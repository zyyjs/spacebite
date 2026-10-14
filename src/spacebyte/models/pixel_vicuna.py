"""Official EVA-CLIP-G/Image Q-Former connected to the frozen Vicuna path."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor, nn

from .byte_qformer_bridge import (
    DEFAULT_VOCAB_SIZE,
    LLM_HIDDEN_DIM,
    QFORMER_HIDDEN_DIM,
    QFORMER_INTERMEDIATE_DIM,
    QFORMER_NUM_HEADS,
    QFORMER_NUM_LAYERS,
    QUERY_TOKEN_COUNT,
    _default_qformer_source,
    _load_lavis_qformer,
)
from .byte_vicuna import ByteVicuna, ByteVicunaInputs


EVA_VISION_WIDTH = 1408
EVA_IMAGE_SIZE = 224
EVA_PATCH_SIZE = 14
INSTRUCTBLIP_CHECKPOINT_URL = (
    "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/"
    "InstructBLIP/instruct_blip_vicuna7b_trimmed.pth"
)
EVA_CHECKPOINT_URL = (
    "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/"
    "BLIP2/eva_vit_g.pth"
)


def _default_eva_source() -> Path:
    return Path(__file__).resolve().parents[3] / "lavis" / "lavis" / "models" / "eva_vit.py"


@lru_cache(maxsize=None)
def _load_lavis_eva_source(source_path: str):
    """Load pinned EVA source without importing LAVIS's unrelated model registry."""

    path = Path(source_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pinned LAVIS EVA source not found: {path}")
    identifier = hashlib.sha256(str(path).encode()).hexdigest()[:12]
    name = f"_spacebyte_pinned_lavis_eva_{identifier}"
    if name in sys.modules:
        return sys.modules[name]

    # eva_vit.py only needs this function for its convenience downloader. This
    # project supplies an audited local checkpoint and deliberately disables
    # that implicit network path while executing the unmodified pinned source.
    module_names = ("lavis", "lavis.common", "lavis.common.dist_utils")
    previous = {key: sys.modules.get(key) for key in module_names}
    lavis_stub = types.ModuleType("lavis")
    common_stub = types.ModuleType("lavis.common")
    dist_stub = types.ModuleType("lavis.common.dist_utils")

    def _implicit_download_disabled(*_args, **_kwargs):
        raise RuntimeError("Implicit EVA download is disabled; pass an audited checkpoint")

    dist_stub.download_cached_file = _implicit_download_disabled
    sys.modules["lavis"] = lavis_stub
    sys.modules["lavis.common"] = common_stub
    sys.modules["lavis.common.dist_utils"] = dist_stub
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load pinned EVA source: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    finally:
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module


def _tensor_count(values: Mapping[str, Tensor]) -> dict[str, int]:
    return {
        "tensors": len(values),
        "parameters": sum(int(value.numel()) for value in values.values()),
    }


def _audited_tensor_transfer(
    target: nn.Module,
    source: Mapping[str, Tensor],
    key_mapper: Callable[[str], str | None],
) -> dict[str, Any]:
    """Copy only explicitly mapped, exactly shaped tensors and verify equality."""

    target_state = target.state_dict()
    loadable: dict[str, tuple[str, Tensor]] = {}
    checkpoint_only: dict[str, Tensor] = {}
    shape_mismatch: dict[str, dict[str, Any]] = {}
    for source_key, tensor in source.items():
        if not torch.is_tensor(tensor):
            continue
        target_key = key_mapper(source_key)
        if target_key is None or target_key not in target_state:
            checkpoint_only[source_key] = tensor
        elif target_state[target_key].shape != tensor.shape:
            shape_mismatch[source_key] = {
                "target_key": target_key,
                "source_shape": list(tensor.shape),
                "target_shape": list(target_state[target_key].shape),
                "parameters": int(tensor.numel()),
            }
        elif target_key in loadable:
            raise RuntimeError(f"Multiple source tensors map to {target_key}")
        else:
            loadable[target_key] = (source_key, tensor)

    with torch.no_grad():
        for target_key, (_source_key, tensor) in loadable.items():
            target_state[target_key].copy_(tensor)
    failed_verification = []
    for target_key, (source_key, tensor) in loadable.items():
        if not torch.equal(target.state_dict()[target_key].cpu(), tensor.cpu()):
            failed_verification.append({"source": source_key, "target": target_key})
    if failed_verification:
        raise RuntimeError(f"Post-load tensor verification failed: {failed_verification[:5]}")

    model_only = {
        key: value for key, value in target_state.items() if key not in loadable
    }
    loaded_tensors = {key: value[1] for key, value in loadable.items()}
    mismatch_tensors = {
        key: source[key] for key in shape_mismatch
    }
    return {
        "loaded": _tensor_count(loaded_tensors),
        "checkpoint_only": _tensor_count(checkpoint_only),
        "model_only": _tensor_count(model_only),
        "shape_mismatch": _tensor_count(mismatch_tensors),
        "loaded_keys": [
            {"source": source_key, "target": target_key}
            for target_key, (source_key, _tensor) in sorted(loadable.items())
        ],
        "checkpoint_only_keys": sorted(checkpoint_only),
        "model_only_keys": sorted(model_only),
        "shape_mismatches": shape_mismatch,
        "post_load_exact_match": not failed_verification,
    }


def build_official_eva_vit_g(
    checkpoint_path: str | Path,
    *,
    source_path: str | Path | None = None,
    mixed_fp16: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Construct pinned EVA-CLIP-G and audit its official checkpoint load."""

    source = Path(source_path or _default_eva_source()).resolve()
    module = _load_lavis_eva_source(str(source))
    model = module.VisionTransformer(
        img_size=EVA_IMAGE_SIZE,
        patch_size=EVA_PATCH_SIZE,
        use_mean_pooling=False,
        embed_dim=EVA_VISION_WIDTH,
        depth=39,
        num_heads=EVA_VISION_WIDTH // 88,
        mlp_ratio=4.3637,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=module.partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint=False,
    )
    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=True, mmap=True
    )
    if not isinstance(checkpoint, Mapping) or not all(
        torch.is_tensor(value) for value in checkpoint.values()
    ):
        raise RuntimeError("EVA checkpoint must be a flat tensor state_dict")
    audit = _audited_tensor_transfer(model, checkpoint, lambda key: key)
    if audit["shape_mismatch"]["tensors"]:
        raise RuntimeError("Official EVA checkpoint contains shape mismatches")
    # Official LAVIS converts only convolutional and linear weights to FP16.
    # The opt-out exists solely for controlled FP32 cache diagnostics.
    if mixed_fp16:
        module.convert_weights_to_fp16(model)
    model.requires_grad_(False)
    model.eval()
    model.train = lambda mode=True: model
    audit.update(
        {
            "checkpoint_path": str(Path(checkpoint_path).resolve()),
            "source_path": str(source),
            "vision_width": int(model.num_features),
            "image_size": EVA_IMAGE_SIZE,
            "patch_size": EVA_PATCH_SIZE,
            "official_mixed_fp16_conversion": mixed_fp16,
        }
    )
    return model, audit


@dataclass(frozen=True)
class ImageQFormerOutput:
    query_features: Tensor
    projected_features: Tensor
    query_attention_mask: Tensor
    combined_attention_mask: Tensor


class ImageQFormerBridge(nn.Module):
    """Official 1408-wide Image Q-Former and 768-to-4096 projection."""

    def __init__(
        self,
        *,
        encoder_width: int = EVA_VISION_WIDTH,
        hidden_size: int = QFORMER_HIDDEN_DIM,
        num_query_tokens: int = QUERY_TOKEN_COUNT,
        cross_attention_freq: int = 2,
        llm_hidden_size: int = LLM_HIDDEN_DIM,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        qformer_source: Path | None = None,
    ) -> None:
        super().__init__()
        module = _load_lavis_qformer(str((qformer_source or _default_qformer_source()).resolve()))
        config = module.BertConfig(
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
        self.vision_layernorm = nn.LayerNorm(encoder_width)
        self.qformer = module.BertLMHeadModel(config=config)
        self.qformer.cls = None
        self.query_tokens = nn.Parameter(torch.empty(1, num_query_tokens, hidden_size))
        nn.init.normal_(self.query_tokens, mean=0.0, std=config.initializer_range)
        self.llm_projector = nn.Linear(hidden_size, llm_hidden_size)
        self.encoder_width = encoder_width
        self.num_query_tokens = num_query_tokens
        self.vocab_size = vocab_size

    def forward(
        self,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor,
        input_ids: Tensor,
        text_attention_mask: Tensor,
    ) -> ImageQFormerOutput:
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[-1] != self.encoder_width:
            raise ValueError(f"vision features must be [B,T,{self.encoder_width}]")
        if encoder_attention_mask.shape != encoder_hidden_states.shape[:2]:
            raise ValueError("vision attention mask must have shape [B,T]")
        if input_ids.shape != text_attention_mask.shape:
            raise ValueError("instruction ids and mask must have equal shape")
        if input_ids.shape[0] != encoder_hidden_states.shape[0]:
            raise ValueError("vision and text batch sizes differ")
        normalized = self.vision_layernorm(encoder_hidden_states)
        query_tokens = self.query_tokens.expand(encoder_hidden_states.shape[0], -1, -1)
        query_mask = torch.ones(query_tokens.shape[:2], dtype=torch.long, device=query_tokens.device)
        combined_mask = torch.cat([query_mask, text_attention_mask.long()], dim=1)
        output = self.qformer.bert(
            input_ids=input_ids.long(),
            attention_mask=combined_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=normalized,
            encoder_attention_mask=encoder_attention_mask.long(),
            return_dict=True,
        )
        query_features = output.last_hidden_state[:, : self.num_query_tokens]
        return ImageQFormerOutput(
            query_features=query_features,
            projected_features=self.llm_projector(query_features),
            query_attention_mask=query_mask,
            combined_attention_mask=combined_mask,
        )


def initialize_image_bridge_from_instructblip(
    bridge: ImageQFormerBridge, checkpoint_path: str | Path
) -> dict[str, Any]:
    """Audited transfer of the official InstructBLIP image-side tensors."""

    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=True, mmap=True
    )
    if not isinstance(checkpoint, Mapping) or "model" not in checkpoint:
        raise RuntimeError("InstructBLIP checkpoint must contain a model state_dict")
    source = checkpoint["model"]
    if not isinstance(source, Mapping):
        raise RuntimeError("InstructBLIP model entry is not a state_dict")

    def mapping(key: str) -> str | None:
        if key == "query_tokens":
            return "query_tokens"
        if key.startswith("Qformer."):
            return "qformer." + key.removeprefix("Qformer.")
        if key.startswith("ln_vision."):
            return "vision_layernorm." + key.removeprefix("ln_vision.")
        if key.startswith("llm_proj."):
            return "llm_projector." + key.removeprefix("llm_proj.")
        return None

    report = _audited_tensor_transfer(bridge, source, mapping)
    loaded_targets = {item["target"] for item in report["loaded_keys"]}
    expected_targets = set(bridge.state_dict())
    if loaded_targets != expected_targets or report["shape_mismatch"]["tensors"]:
        missing = sorted(expected_targets - loaded_targets)
        raise RuntimeError(
            "Official InstructBLIP image bridge is not fully compatible: "
            f"model_only={missing[:10]} shape_mismatch={report['shape_mismatches']}"
        )
    report.update(
        {
            "checkpoint_path": str(Path(checkpoint_path).resolve()),
            "checkpoint_top_level_keys": sorted(checkpoint),
            "checkpoint_model": _tensor_count(
                {key: value for key, value in source.items() if torch.is_tensor(value)}
            ),
            "mapping_rules": {
                "Qformer.*": "qformer.*",
                "query_tokens": "query_tokens",
                "ln_vision.*": "vision_layernorm.*",
                "llm_proj.*": "llm_projector.*",
            },
        }
    )
    return report


@dataclass
class PixelVicunaInputs:
    pixel_values: Tensor
    vision_features: Tensor
    vision_attention_mask: Tensor
    image_query_features: Tensor
    projected_image_queries: Tensor
    vicuna_inputs_embeds: Tensor
    vicuna_attention_mask: Tensor
    labels: Tensor | None
    query_slice: slice
    prompt_slices: list[slice]
    answer_slices: list[slice]

    @property
    def inputs_embeds(self) -> Tensor:
        return self.vicuna_inputs_embeds

    @property
    def attention_mask(self) -> Tensor:
        return self.vicuna_attention_mask

    @property
    def query_features(self) -> Tensor:
        return self.image_query_features

    @property
    def projected_queries(self) -> Tensor:
        return self.projected_image_queries


@dataclass
class PixelVicunaOutput:
    loss: Tensor
    logits: Tensor
    prepared: PixelVicunaInputs


class PixelVicuna(ByteVicuna):
    """Independent frozen-EVA -> Image Q-Former -> frozen-Vicuna path."""

    CONDITIONS = {"normal", "zero", "shuffled"}

    def __init__(self, *, visual_encoder: nn.Module, bridge: ImageQFormerBridge, **kwargs) -> None:
        super().__init__(byteformer_adapter=visual_encoder, bridge=bridge, **kwargs)

    @property
    def visual_encoder(self) -> nn.Module:
        return self.byteformer_adapter

    def encode_queries(
        self,
        pixel_values: Tensor,
        questions: str | Sequence[str],
        *,
        pixel_condition: str = "normal",
        shuffled_indices: Tensor | None = None,
    ) -> tuple[ImageQFormerOutput, Tensor, Tensor]:
        if pixel_condition not in self.CONDITIONS:
            raise ValueError(f"Unknown Pixel condition: {pixel_condition}")
        if pixel_values.ndim != 4 or pixel_values.shape[1:] != (3, 224, 224):
            raise ValueError("pixel_values must have shape [B,3,224,224]")
        question_batch = self._normalize_text_batch(questions, pixel_values.shape[0], "questions")
        instruction_ids, instruction_mask = self._qformer_tokens(question_batch)
        with torch.no_grad():
            vision_features = self.visual_encoder(pixel_values)
        if pixel_condition == "zero":
            vision_features = torch.zeros_like(vision_features)
        elif pixel_condition == "shuffled":
            if shuffled_indices is None:
                raise ValueError("Shuffled-Pixel requires explicit global shuffled_indices")
            indices = shuffled_indices.to(device=vision_features.device, dtype=torch.long)
            if indices.shape != (vision_features.shape[0],):
                raise ValueError("shuffled_indices must contain one global source index per sample")
            identity = torch.arange(indices.numel(), device=indices.device)
            if indices.numel() < 2 or not bool(indices.ne(identity).all().item()):
                raise ValueError("Shuffled-Pixel must be fixed-point-free and cannot be batch-size one")
            if sorted(indices.cpu().tolist()) != list(range(indices.numel())):
                raise ValueError("shuffled_indices must be a permutation")
            vision_features = vision_features.index_select(0, indices)
        vision_mask = torch.ones(
            vision_features.shape[:2], dtype=torch.long, device=vision_features.device
        )
        bridge_output = self.bridge(
            vision_features,
            vision_mask,
            instruction_ids,
            instruction_mask,
        )
        return bridge_output, vision_features, vision_mask

    @staticmethod
    def _as_pixel_inputs(
        base: ByteVicunaInputs, pixel_values: Tensor
    ) -> PixelVicunaInputs:
        return PixelVicunaInputs(
            pixel_values=pixel_values,
            vision_features=base.byte_features,
            vision_attention_mask=base.byte_attention_mask,
            image_query_features=base.query_features,
            projected_image_queries=base.projected_queries,
            vicuna_inputs_embeds=base.inputs_embeds,
            vicuna_attention_mask=base.attention_mask,
            labels=base.labels,
            query_slice=base.query_slice,
            prompt_slices=base.prompt_slices,
            answer_slices=base.answer_slices,
        )

    def prepare_inputs(
        self,
        pixel_values: Tensor,
        questions: str | Sequence[str],
        answers: str | Sequence[str] | None = None,
        *,
        pixel_condition: str = "normal",
        shuffled_indices: Tensor | None = None,
        projected_query_scale: float = 1.0,
        extra_text_padding: int = 0,
        text_padding_side: str = "right",
        answer_only_loss: bool = False,
    ) -> PixelVicunaInputs:
        question_batch = self._normalize_text_batch(questions, pixel_values.shape[0], "questions")
        bridge_output, features, mask = self.encode_queries(
            pixel_values,
            question_batch,
            pixel_condition=pixel_condition,
            shuffled_indices=shuffled_indices,
        )
        base = self._prepare_from_bridge_output(
            bridge_output,
            features,
            mask,
            question_batch,
            answers,
            projected_query_scale=projected_query_scale,
            extra_text_padding=extra_text_padding,
            text_padding_side=text_padding_side,
            answer_only_loss=answer_only_loss,
        )
        if base.labels is not None:
            ignored = int(base.labels.eq(-100).sum().item())
            supervised = int(base.labels.ne(-100).sum().item())
            if ignored + supervised != base.labels.numel():
                raise RuntimeError("ignored + supervised labels must equal labels.numel()")
        return self._as_pixel_inputs(base, pixel_values)

    def prepare_inputs_from_vision_features(
        self,
        vision_features: Tensor,
        vision_attention_mask: Tensor,
        questions: str | Sequence[str],
        answers: str | Sequence[str] | None = None,
        *,
        pixel_condition: str = "normal",
        projected_query_scale: float = 1.0,
        extra_text_padding: int = 0,
        answer_only_loss: bool = False,
    ) -> PixelVicunaInputs:
        """Build Vicuna inputs from audited frozen-EVA cache tensors."""

        if pixel_condition not in {"normal", "zero"}:
            raise ValueError(
                "Cached Shuffled-Pixel must be implemented by global dataset mapping"
            )
        if (
            vision_features.ndim != 3
            or vision_features.shape[-1] != self.bridge.encoder_width
        ):
            raise ValueError(
                f"vision_features must be [B,T,{self.bridge.encoder_width}]"
            )
        if vision_attention_mask.shape != vision_features.shape[:2]:
            raise ValueError("vision_attention_mask must have shape [B,T]")
        batch_size = vision_features.shape[0]
        question_batch = self._normalize_text_batch(questions, batch_size, "questions")
        instruction_ids, instruction_mask = self._qformer_tokens(question_batch)
        conditioned = (
            torch.zeros_like(vision_features)
            if pixel_condition == "zero"
            else vision_features
        )
        bridge_output = self.bridge(
            conditioned,
            vision_attention_mask,
            instruction_ids,
            instruction_mask,
        )
        base = self._prepare_from_bridge_output(
            bridge_output,
            conditioned,
            vision_attention_mask,
            question_batch,
            answers,
            projected_query_scale=projected_query_scale,
            extra_text_padding=extra_text_padding,
            text_padding_side="right",
            answer_only_loss=answer_only_loss,
        )
        if base.labels is not None:
            ignored = int(base.labels.eq(-100).sum().item())
            supervised = int(base.labels.ne(-100).sum().item())
            if ignored + supervised != base.labels.numel():
                raise RuntimeError("ignored + supervised labels must equal labels.numel()")
        # Cached execution has no live RGB tensor; retain an empty, typed marker
        # rather than fabricating pixels that never entered the model.
        pixel_marker = vision_features.new_empty((batch_size, 0, 0, 0))
        return self._as_pixel_inputs(base, pixel_marker)

    def forward(
        self,
        pixel_values: Tensor,
        questions: str | Sequence[str],
        answers: str | Sequence[str],
        **kwargs,
    ) -> PixelVicunaOutput:
        prepared = self.prepare_inputs(pixel_values, questions, answers, **kwargs)
        outputs = self.vicuna(
            inputs_embeds=prepared.vicuna_inputs_embeds,
            attention_mask=prepared.vicuna_attention_mask,
            labels=prepared.labels,
            use_cache=False,
            return_dict=True,
        )
        return PixelVicunaOutput(outputs.loss, outputs.logits, prepared)

    def forward_from_vision_features(
        self,
        vision_features: Tensor,
        vision_attention_mask: Tensor,
        questions: str | Sequence[str],
        answers: str | Sequence[str],
        **kwargs,
    ) -> PixelVicunaOutput:
        """Differentiable answer-loss path over frozen-EVA cached features."""

        prepared = self.prepare_inputs_from_vision_features(
            vision_features,
            vision_attention_mask,
            questions,
            answers,
            **kwargs,
        )
        outputs = self.vicuna(
            inputs_embeds=prepared.vicuna_inputs_embeds,
            attention_mask=prepared.vicuna_attention_mask,
            labels=prepared.labels,
            use_cache=False,
            return_dict=True,
        )
        return PixelVicunaOutput(outputs.loss, outputs.logits, prepared)


def parameter_summary(model: PixelVicuna) -> dict[str, dict[str, int]]:
    groups = {
        "visual_encoder": model.visual_encoder,
        "image_qformer_bridge": model.bridge,
        "vicuna": model.vicuna,
    }
    return {
        name: {
            "total": sum(parameter.numel() for parameter in module.parameters()),
            "trainable": sum(
                parameter.numel() for parameter in module.parameters() if parameter.requires_grad
            ),
        }
        for name, module in groups.items()
    }
