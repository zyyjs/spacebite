"""Instruction-aware Q-Former bridge for frozen ByteFormer token features."""

from __future__ import annotations

import importlib.util
import hashlib
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import torch
from torch import Tensor, nn


BYTE_FEATURE_DIM = 192
QFORMER_HIDDEN_DIM = 768
QFORMER_INTERMEDIATE_DIM = 3072
QFORMER_NUM_HEADS = 12
QFORMER_NUM_LAYERS = 12
QUERY_TOKEN_COUNT = 32
LLM_HIDDEN_DIM = 4096
DEFAULT_VOCAB_SIZE = 30523


def _default_qformer_source() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "lavis"
        / "lavis"
        / "models"
        / "blip2_models"
        / "Qformer.py"
    )


@lru_cache(maxsize=None)
def _load_lavis_qformer(source_path: str) -> ModuleType:
    """Load only the pinned LAVIS Qformer.py, without importing all of LAVIS."""
    path = Path(source_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LAVIS Q-Former source not found: {path}")
    path_id = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
    module_name = f"_spacebyte_pinned_lavis_qformer_{path_id}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot construct an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


@dataclass(frozen=True)
class ByteQFormerBridgeOutput:
    """Outputs produced before any Vicuna model is introduced."""

    query_features: Tensor
    projected_features: Tensor
    query_attention_mask: Tensor
    combined_attention_mask: Tensor


class ByteQFormerBridge(nn.Module):
    """Map ByteFormer tokens and an instruction to 32 Vicuna-width tokens.

    This module is intentionally random-initialized. It loads the Q-Former
    *implementation* from the pinned LAVIS submodule, but never loads LAVIS,
    BLIP-2, X-InstructBLIP, or Vicuna model weights.
    """

    def __init__(
        self,
        *,
        encoder_width: int = BYTE_FEATURE_DIM,
        hidden_size: int = QFORMER_HIDDEN_DIM,
        num_query_tokens: int = QUERY_TOKEN_COUNT,
        cross_attention_freq: int = 2,
        llm_hidden_size: int = LLM_HIDDEN_DIM,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        qformer_source: Path | None = None,
    ) -> None:
        super().__init__()
        if (
            min(
                encoder_width,
                hidden_size,
                num_query_tokens,
                cross_attention_freq,
                llm_hidden_size,
                vocab_size,
            )
            <= 0
        ):
            raise ValueError("All bridge dimensions and frequencies must be positive")
        if hidden_size % QFORMER_NUM_HEADS:
            raise ValueError("hidden_size must be divisible by 12 attention heads")

        source = (qformer_source or _default_qformer_source()).resolve()
        qformer_module = _load_lavis_qformer(str(source))
        config = qformer_module.BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=QFORMER_NUM_LAYERS,
            num_attention_heads=QFORMER_NUM_HEADS,
            intermediate_size=QFORMER_INTERMEDIATE_DIM,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            is_decoder=True,
            add_cross_attention=True,
        )
        config.encoder_width = encoder_width
        config.cross_attention_freq = cross_attention_freq
        config.query_length = num_query_tokens

        self.byte_layernorm = nn.LayerNorm(encoder_width)
        self.qformer = qformer_module.BertLMHeadModel(config=config)
        # The bridge consumes hidden states directly. Keeping the MLM head
        # would add unrelated trainable parameters that never receive gradients.
        self.qformer.cls = None
        self.query_tokens = nn.Parameter(torch.empty(1, num_query_tokens, hidden_size))
        nn.init.normal_(self.query_tokens, mean=0.0, std=config.initializer_range)
        self.llm_projector = nn.Linear(hidden_size, llm_hidden_size)

        self.initialization_mode = "random"
        self.qformer_source = str(source)
        self.vocab_size = vocab_size
        self.encoder_width = encoder_width
        self.num_query_tokens = num_query_tokens

    @property
    def config(self):
        return self.qformer.config

    def _validate_inputs(
        self,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor,
        input_ids: Tensor,
        text_attention_mask: Tensor,
    ) -> None:
        if (
            encoder_hidden_states.ndim != 3
            or encoder_hidden_states.shape[-1] != self.encoder_width
        ):
            raise ValueError(
                f"encoder_hidden_states must have shape [B, T, {self.encoder_width}], "
                f"got {tuple(encoder_hidden_states.shape)}"
            )
        batch_size = encoder_hidden_states.shape[0]
        if encoder_attention_mask.shape != encoder_hidden_states.shape[:2]:
            raise ValueError("encoder_attention_mask must have shape [B, T]")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, L]")
        if input_ids.shape[0] != batch_size:
            raise ValueError("Byte and instruction batch sizes must match")
        if text_attention_mask.shape != input_ids.shape:
            raise ValueError("text_attention_mask must have shape [B, L]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must be int32 or int64")
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("Instruction exceeds Q-Former's position embedding limit")
        if bool(((input_ids < 0) | (input_ids >= self.vocab_size)).any().item()):
            raise ValueError("instruction_input_ids contain an out-of-vocabulary id")
        for name, mask in (
            ("encoder_attention_mask", encoder_attention_mask),
            ("text_attention_mask", text_attention_mask),
        ):
            if not bool(((mask == 0) | (mask == 1)).all().item()):
                raise ValueError(f"{name} values must be 0 or 1")

    def forward(
        self,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor,
        input_ids: Tensor,
        text_attention_mask: Tensor,
    ) -> ByteQFormerBridgeOutput:
        self._validate_inputs(
            encoder_hidden_states,
            encoder_attention_mask,
            input_ids,
            text_attention_mask,
        )
        batch_size = encoder_hidden_states.shape[0]
        device = encoder_hidden_states.device
        if input_ids.device != device:
            raise ValueError("All bridge inputs must be on the same device")

        normalized_byte_features = self.byte_layernorm(encoder_hidden_states)
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        query_attention_mask = torch.ones(
            (batch_size, self.num_query_tokens), dtype=torch.long, device=device
        )
        text_attention_mask = text_attention_mask.to(device=device, dtype=torch.long)
        combined_attention_mask = torch.cat(
            [query_attention_mask, text_attention_mask], dim=1
        )

        outputs = self.qformer.bert(
            input_ids=input_ids.to(dtype=torch.long),
            attention_mask=combined_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=normalized_byte_features,
            encoder_attention_mask=encoder_attention_mask.to(
                device=device, dtype=torch.long
            ),
            return_dict=True,
        )
        query_features = outputs.last_hidden_state[:, : self.num_query_tokens, :]
        projected_features = self.llm_projector(query_features)
        return ByteQFormerBridgeOutput(
            query_features=query_features,
            projected_features=projected_features,
            query_attention_mask=query_attention_mask,
            combined_attention_mask=combined_attention_mask,
        )


def freeze_byteformer(byteformer: nn.Module) -> nn.Module:
    """Freeze a ByteFormer/adapter tree and put every submodule in eval mode."""
    byteformer.requires_grad_(False)
    byteformer.eval()
    return byteformer
