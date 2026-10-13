"""Frozen Vicuna language model driven by ByteFormer/Q-Former query tokens."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from .byte_qformer_bridge import ByteQFormerBridge, ByteQFormerBridgeOutput, freeze_byteformer


VICUNA_REPO_ID = "lmsys/vicuna-7b-v1.1"
VICUNA_REVISION = "694d39416b1744d939760f5257c4bb08b2fae400"
VICUNA_HIDDEN_SIZE = 4096
VICUNA_ORIGINAL_VOCAB_SIZE = 32000
XINSTRUCT_PREFIX = "USER: "
XINSTRUCT_IMAGE_CUE = "image: "
XINSTRUCT_POSTFIX = "\nASSISTANT:"


def scale_projected_queries(projected: Tensor, scale: float) -> Tensor:
    """Apply the explicit Byte-prefix ablation without changing other inputs."""

    return projected * scale


@dataclass
class ByteVicunaInputs:
    """Auditable input bundle immediately before the frozen Vicuna model."""

    inputs_embeds: Tensor
    attention_mask: Tensor
    labels: Tensor | None
    query_features: Tensor
    projected_queries: Tensor
    query_slice: slice
    prompt_slices: list[slice]
    answer_slices: list[slice]
    byte_features: Tensor
    byte_attention_mask: Tensor


@dataclass
class ByteVicunaOutput:
    """Loss-path outputs plus the exact tensors supplied to Vicuna."""

    loss: Tensor
    logits: Tensor
    prepared: ByteVicunaInputs


def build_xinstruct_vicuna_tokenizer(model_path: str | Path):
    """Reproduce the tokenizer mutation in pinned X-InstructBLIP.

    X-InstructBLIP adds a new ``[PAD]`` row, then aliases BOS, EOS, and UNK to
    ``</s>``. It does not reuse EOS as padding.
    """

    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained(
        str(Path(model_path).resolve()),
        use_fast=False,
        truncation_side="left",
        local_files_only=True,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.bos_token = "</s>"
    tokenizer.eos_token = "</s>"
    tokenizer.unk_token = "</s>"
    return tokenizer


def build_xinstruct_qformer_tokenizer(tokenizer_path: str | Path):
    """Build the BERT tokenizer used for Q-Former instruction tokens."""

    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(
        str(Path(tokenizer_path).resolve()),
        truncation_side="right",
        local_files_only=True,
    )
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    return tokenizer


def load_frozen_vicuna(
    model_path: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype = torch.float16,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    """Load local official Vicuna weights and freeze every LLM parameter."""

    from transformers import LlamaForCausalLM

    resolved = Path(model_path).resolve()
    tokenizer = build_xinstruct_vicuna_tokenizer(resolved)
    model = LlamaForCausalLM.from_pretrained(
        str(resolved),
        torch_dtype=dtype,
        local_files_only=True,
    )
    original_vocab_size = int(model.config.vocab_size)
    original_special_token_ids = {
        "bos_token_id": model.config.bos_token_id,
        "eos_token_id": model.config.eos_token_id,
        "pad_token_id": model.config.pad_token_id,
    }
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.use_cache = False
    model.to(device)
    model.requires_grad_(False)
    model.eval()

    if model.config.hidden_size != VICUNA_HIDDEN_SIZE:
        raise RuntimeError(
            f"Expected Vicuna hidden size {VICUNA_HIDDEN_SIZE}, "
            f"got {model.config.hidden_size}"
        )
    if original_vocab_size != VICUNA_ORIGINAL_VOCAB_SIZE:
        raise RuntimeError(
            f"Expected original vocabulary {VICUNA_ORIGINAL_VOCAB_SIZE}, "
            f"got {original_vocab_size}"
        )
    report = {
        "model_path": str(resolved),
        "model_type": model.config.model_type,
        "architectures": list(model.config.architectures or []),
        "hidden_size": model.config.hidden_size,
        "intermediate_size": model.config.intermediate_size,
        "original_vocab_size": original_vocab_size,
        "checkpoint_special_token_ids": original_special_token_ids,
        "resized_vocab_size": len(tokenizer),
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "max_position_embeddings": model.config.max_position_embeddings,
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "unk_token": tokenizer.unk_token,
        "unk_token_id": tokenizer.unk_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "dtype": str(dtype),
        "all_parameters_frozen": not any(p.requires_grad for p in model.parameters()),
    }
    return model, tokenizer, report


class ByteVicuna(nn.Module):
    """Full ByteFormer -> Byte Q-Former -> frozen Vicuna model.

    The prefix layout follows the pinned X-InstructBLIP generation path:
    ``USER:`` tokens, ``image:`` cue tokens without their duplicate start
    token, 32 projected query tokens, then question and ``ASSISTANT:`` tokens.
    Only answer tokens receive causal-LM labels.
    """

    def __init__(
        self,
        *,
        byteformer_adapter: nn.Module,
        bridge: ByteQFormerBridge,
        vicuna: nn.Module,
        llm_tokenizer: Any,
        qformer_tokenizer: Any,
        max_instruction_length: int = 128,
        max_answer_length: int = 256,
    ) -> None:
        super().__init__()
        self.byteformer_adapter = freeze_byteformer(byteformer_adapter)
        self.bridge = bridge
        self.vicuna = vicuna
        self.llm_tokenizer = llm_tokenizer
        self.qformer_tokenizer = qformer_tokenizer
        self.max_instruction_length = max_instruction_length
        self.max_answer_length = max_answer_length

        hidden_size = int(vicuna.config.hidden_size)
        if bridge.llm_projector.out_features != hidden_size:
            raise ValueError(
                "Q-Former projector output does not equal Vicuna hidden size: "
                f"{bridge.llm_projector.out_features} != {hidden_size}"
            )
        self.vicuna.requires_grad_(False)
        self.vicuna.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen components must remain deterministic even while the bridge is
        # in training mode. Gradients still propagate through Vicuna operations
        # to inputs_embeds because this does not use torch.no_grad().
        self.byteformer_adapter.eval()
        self.vicuna.eval()
        self.bridge.train(mode)
        return self

    @property
    def device(self) -> torch.device:
        return next(self.bridge.parameters()).device

    def _normalize_text_batch(
        self, values: str | Sequence[str], batch_size: int, name: str
    ) -> list[str]:
        if isinstance(values, str):
            result = [values] * batch_size
        else:
            result = list(values)
        if len(result) != batch_size or not all(isinstance(item, str) for item in result):
            raise ValueError(f"{name} must contain one string per batch element")
        return result

    def _qformer_tokens(self, questions: list[str]) -> tuple[Tensor, Tensor]:
        tokens = self.qformer_tokenizer(
            questions,
            padding="longest",
            truncation=True,
            max_length=self.max_instruction_length,
            return_tensors="pt",
        )
        return (
            tokens.input_ids.to(self.device),
            tokens.attention_mask.to(self.device),
        )

    def encode_queries(
        self,
        byte_tensor: Tensor,
        questions: str | Sequence[str],
    ) -> tuple[ByteQFormerBridgeOutput, Tensor, Tensor]:
        """Encode JPEG bytes while retaining gradients only through the bridge."""

        if byte_tensor.device != self.device:
            raise ValueError("byte_tensor and Byte Q-Former must be on the same device")
        batch_size = byte_tensor.shape[0]
        question_batch = self._normalize_text_batch(questions, batch_size, "questions")
        instruction_ids, instruction_mask = self._qformer_tokens(question_batch)
        with torch.no_grad():
            byte_features, byte_mask = self.byteformer_adapter(byte_tensor)
        bridge_output = self.bridge(
            encoder_hidden_states=byte_features,
            encoder_attention_mask=byte_mask,
            input_ids=instruction_ids,
            text_attention_mask=instruction_mask,
        )
        return bridge_output, byte_features, byte_mask

    def _token_ids(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool,
        truncation_side: str,
        max_length: int,
    ) -> tuple[Tensor, Tensor]:
        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = truncation_side
        tokens = self.llm_tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )
        return tokens.input_ids.to(self.device), tokens.attention_mask.to(self.device)

    def _fixed_text_prefix(self, batch_size: int) -> tuple[Tensor, Tensor]:
        # Prefix includes the single X-InstructBLIP start token. The image cue
        # is tokenized independently and therefore has its duplicate start
        # token removed when clean_tokenization=True.
        prefix_ids, prefix_mask = self._token_ids(
            [XINSTRUCT_PREFIX] * batch_size,
            add_special_tokens=True,
            truncation_side="left",
            max_length=32,
        )
        cue_ids, cue_mask = self._token_ids(
            [XINSTRUCT_IMAGE_CUE] * batch_size,
            add_special_tokens=True,
            truncation_side="left",
            max_length=32,
        )
        return (
            torch.cat([prefix_ids, cue_ids[:, 1:]], dim=1),
            torch.cat([prefix_mask, cue_mask[:, 1:]], dim=1),
        )

    def prepare_inputs(
        self,
        byte_tensor: Tensor,
        questions: str | Sequence[str],
        answers: str | Sequence[str] | None = None,
        *,
        projected_query_scale: float = 1.0,
        extra_text_padding: int = 0,
        text_padding_side: str = "right",
    ) -> ByteVicunaInputs:
        """Build exact Vicuna embeddings, masks, and optional answer labels."""

        if extra_text_padding < 0:
            raise ValueError("extra_text_padding must be non-negative")
        batch_size = byte_tensor.shape[0]
        question_batch = self._normalize_text_batch(questions, batch_size, "questions")
        bridge_output, byte_features, byte_mask = self.encode_queries(
            byte_tensor, question_batch
        )
        return self._prepare_from_bridge_output(
            bridge_output,
            byte_features,
            byte_mask,
            question_batch,
            answers,
            projected_query_scale=projected_query_scale,
            extra_text_padding=extra_text_padding,
            text_padding_side=text_padding_side,
        )

    def prepare_inputs_from_byte_features(
        self,
        byte_features: Tensor,
        byte_attention_mask: Tensor,
        questions: str | Sequence[str],
        answers: str | Sequence[str] | None = None,
        *,
        projected_query_scale: float = 1.0,
        extra_text_padding: int = 0,
        text_padding_side: str = "right",
    ) -> ByteVicunaInputs:
        """Prepare Vicuna inputs from masked ByteFormer tokens for mask audits."""

        if extra_text_padding < 0:
            raise ValueError("extra_text_padding must be non-negative")
        batch_size = byte_features.shape[0]
        question_batch = self._normalize_text_batch(questions, batch_size, "questions")
        instruction_ids, instruction_mask = self._qformer_tokens(question_batch)
        bridge_output = self.bridge(
            encoder_hidden_states=byte_features,
            encoder_attention_mask=byte_attention_mask,
            input_ids=instruction_ids,
            text_attention_mask=instruction_mask,
        )
        return self._prepare_from_bridge_output(
            bridge_output,
            byte_features,
            byte_attention_mask,
            question_batch,
            answers,
            projected_query_scale=projected_query_scale,
            extra_text_padding=extra_text_padding,
            text_padding_side=text_padding_side,
        )

    def _prepare_from_bridge_output(
        self,
        bridge_output: ByteQFormerBridgeOutput,
        byte_features: Tensor,
        byte_mask: Tensor,
        question_batch: list[str],
        answers: str | Sequence[str] | None,
        *,
        projected_query_scale: float,
        extra_text_padding: int,
        text_padding_side: str,
    ) -> ByteVicunaInputs:
        if text_padding_side not in {"left", "right"}:
            raise ValueError("text_padding_side must be 'left' or 'right'")
        if answers is not None and text_padding_side != "right":
            raise ValueError("The X-InstructBLIP loss path uses right padding")
        batch_size = byte_features.shape[0]
        projected = scale_projected_queries(
            bridge_output.projected_features, projected_query_scale
        )

        fixed_ids, fixed_mask = self._fixed_text_prefix(batch_size)
        prompt_text = [f"{question.strip()}{XINSTRUCT_POSTFIX}" for question in question_batch]
        prompt_ids, prompt_mask = self._token_ids(
            prompt_text,
            add_special_tokens=False,
            truncation_side="left",
            max_length=self.max_instruction_length,
        )

        answer_ids = answer_mask = None
        answer_batch: list[str] | None = None
        if answers is not None:
            answer_batch = self._normalize_text_batch(answers, batch_size, "answers")
            answer_ids, answer_mask = self._token_ids(
                [f"{answer}{self.llm_tokenizer.eos_token}" for answer in answer_batch],
                add_special_tokens=True,
                truncation_side="right",
                max_length=self.max_answer_length,
            )

        # X-InstructBLIP inserts output_ids[1:] after each unpadded prompt. Build
        # each row explicitly so labels and variable-length padding are auditable.
        text_rows: list[Tensor] = []
        label_rows: list[Tensor] = []
        prompt_lengths: list[int] = []
        answer_lengths: list[int] = []
        for index in range(batch_size):
            prompt_length = int(prompt_mask[index].sum().item())
            prompt_row = prompt_ids[index, :prompt_length]
            prompt_lengths.append(prompt_length)
            if answer_ids is None or answer_mask is None:
                answer_row = prompt_row.new_empty((0,))
            else:
                raw_answer_length = int(answer_mask[index].sum().item())
                # Drop the answer tokenizer's independently inserted start token.
                answer_row = answer_ids[index, 1:raw_answer_length]
            answer_lengths.append(int(answer_row.numel()))
            text_rows.append(torch.cat([prompt_row, answer_row]))
            label_rows.append(
                torch.cat(
                    [
                        torch.full_like(prompt_row, -100),
                        answer_row,
                    ]
                )
            )

        text_width = max(row.numel() for row in text_rows) + extra_text_padding
        pad_id = int(self.llm_tokenizer.pad_token_id)
        text_ids = torch.full(
            (batch_size, text_width), pad_id, dtype=torch.long, device=self.device
        )
        text_mask = torch.zeros(
            (batch_size, text_width), dtype=torch.long, device=self.device
        )
        text_labels = torch.full(
            (batch_size, text_width), -100, dtype=torch.long, device=self.device
        )
        row_offsets: list[int] = []
        for index, (row, labels) in enumerate(zip(text_rows, label_rows)):
            width = row.numel()
            offset = 0 if text_padding_side == "right" else text_width - width
            row_offsets.append(offset)
            text_ids[index, offset : offset + width] = row
            text_mask[index, offset : offset + width] = 1
            text_labels[index, offset : offset + width] = labels

        embedding = self.vicuna.get_input_embeddings()
        embedding_dtype = embedding.weight.dtype
        fixed_embeds = embedding(fixed_ids).to(dtype=embedding_dtype)
        projected_for_llm = projected.to(dtype=embedding_dtype)
        text_embeds = embedding(text_ids).to(dtype=embedding_dtype)
        query_mask = torch.ones(
            projected.shape[:2], dtype=torch.long, device=self.device
        )
        inputs_embeds = torch.cat(
            [fixed_embeds, projected_for_llm, text_embeds], dim=1
        )
        attention_mask = torch.cat([fixed_mask, query_mask, text_mask], dim=1)

        fixed_width = fixed_ids.shape[1]
        query_slice = slice(fixed_width, fixed_width + projected.shape[1])
        labels = None
        if answers is not None:
            prefix_labels = torch.full(
                (batch_size, fixed_width + projected.shape[1]),
                -100,
                dtype=torch.long,
                device=self.device,
            )
            labels = torch.cat([prefix_labels, text_labels], dim=1)

        prompt_slices = []
        answer_slices = []
        text_start = query_slice.stop
        for offset, prompt_length, answer_length in zip(
            row_offsets, prompt_lengths, answer_lengths
        ):
            prompt_start = text_start + offset
            prompt_slices.append(
                slice(prompt_start, prompt_start + prompt_length)
            )
            answer_slices.append(
                slice(
                    prompt_start + prompt_length,
                    prompt_start + prompt_length + answer_length,
                )
            )
        if attention_mask.shape != inputs_embeds.shape[:2]:
            raise RuntimeError("Vicuna attention mask and embeddings diverged")
        if labels is not None and labels.shape != attention_mask.shape:
            raise RuntimeError("Vicuna labels and attention mask diverged")
        return ByteVicunaInputs(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            query_features=bridge_output.query_features,
            projected_queries=projected,
            query_slice=query_slice,
            prompt_slices=prompt_slices,
            answer_slices=answer_slices,
            byte_features=byte_features,
            byte_attention_mask=byte_mask,
        )

    def forward(
        self,
        byte_tensor: Tensor,
        questions: str | Sequence[str],
        answers: str | Sequence[str],
        *,
        projected_query_scale: float = 1.0,
        extra_text_padding: int = 0,
    ) -> ByteVicunaOutput:
        prepared = self.prepare_inputs(
            byte_tensor,
            questions,
            answers,
            projected_query_scale=projected_query_scale,
            extra_text_padding=extra_text_padding,
        )
        outputs = self.vicuna(
            inputs_embeds=prepared.inputs_embeds,
            attention_mask=prepared.attention_mask,
            labels=prepared.labels,
            use_cache=False,
            return_dict=True,
        )
        return ByteVicunaOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            prepared=prepared,
        )

    def forward_from_byte_features(
        self,
        byte_features: Tensor,
        byte_attention_mask: Tensor,
        questions: str | Sequence[str],
        answers: str | Sequence[str],
        *,
        projected_query_scale: float = 1.0,
        extra_text_padding: int = 0,
    ) -> ByteVicunaOutput:
        """Differentiable loss path using cached frozen-ByteFormer outputs."""

        prepared = self.prepare_inputs_from_byte_features(
            byte_features,
            byte_attention_mask,
            questions,
            answers,
            projected_query_scale=projected_query_scale,
            extra_text_padding=extra_text_padding,
        )
        outputs = self.vicuna(
            inputs_embeds=prepared.inputs_embeds,
            attention_mask=prepared.attention_mask,
            labels=prepared.labels,
            use_cache=False,
            return_dict=True,
        )
        return ByteVicunaOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            prepared=prepared,
        )

    def _generate_prepared(
        self,
        prepared: ByteVicunaInputs,
        *,
        do_sample: bool,
        num_beams: int,
        max_length: int,
        min_length: int,
        top_p: float,
        temperature: float,
        repetition_penalty: float,
        length_penalty: float,
        num_return_sequences: int,
    ) -> tuple[list[str], Tensor, ByteVicunaInputs]:
        generated = self.vicuna.generate(
            inputs_embeds=prepared.inputs_embeds,
            attention_mask=prepared.attention_mask,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
            use_cache=False,
            pad_token_id=self.llm_tokenizer.pad_token_id,
            eos_token_id=self.llm_tokenizer.eos_token_id,
        )
        generated = generated.masked_fill(
            generated.eq(0), self.llm_tokenizer.eos_token_id
        )
        text = [
            item.strip()
            for item in self.llm_tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )
        ]
        return text, generated, prepared

    @torch.no_grad()
    def generate(
        self,
        byte_tensor: Tensor,
        questions: str | Sequence[str],
        *,
        do_sample: bool = False,
        num_beams: int = 5,
        max_length: int = 30,
        min_length: int = 1,
        top_p: float = 0.9,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        num_return_sequences: int = 1,
    ) -> tuple[list[str], Tensor, ByteVicunaInputs]:
        self.eval()
        prepared = self.prepare_inputs(
            byte_tensor, questions, text_padding_side="left"
        )
        return self._generate_prepared(
            prepared,
            do_sample=do_sample,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
        )

    @torch.no_grad()
    def generate_from_byte_features(
        self,
        byte_features: Tensor,
        byte_attention_mask: Tensor,
        questions: str | Sequence[str],
        *,
        do_sample: bool = False,
        num_beams: int = 5,
        max_length: int = 30,
        min_length: int = 1,
        top_p: float = 0.9,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        num_return_sequences: int = 1,
        projected_query_scale: float = 1.0,
    ) -> tuple[list[str], Tensor, ByteVicunaInputs]:
        """Generate from cached frozen-ByteFormer outputs."""

        self.eval()
        prepared = self.prepare_inputs_from_byte_features(
            byte_features,
            byte_attention_mask,
            questions,
            projected_query_scale=projected_query_scale,
            text_padding_side="left",
        )
        return self._generate_prepared(
            prepared,
            do_sample=do_sample,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
        )


def embedding_statistics(values: Tensor) -> dict[str, float]:
    """Return the requested scale diagnostics without changing the values."""

    fp32 = values.detach().float()
    return {
        "mean": float(fp32.mean().item()),
        "std": float(fp32.std().item()),
        "mean_l2_norm": float(fp32.norm(dim=-1).mean().item()),
    }
