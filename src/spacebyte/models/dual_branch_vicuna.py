"""Frozen Pixel+Byte query prefixes connected to one frozen Vicuna model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from .byte_qformer_bridge import ByteQFormerBridgeOutput, freeze_byteformer
from .byte_vicuna import ByteVicuna, XINSTRUCT_POSTFIX
from .pixel_vicuna import ImageQFormerOutput


DUAL_CONDITIONS = {
    "pixel_byte",
    "pixel_only",
    "byte_only",
    "zero_both",
    "shuffled_pixel",
    "shuffled_byte",
}


@dataclass
class DualBranchInputs:
    pixel_values: Tensor
    pixel_vision_features: Tensor
    pixel_vision_attention_mask: Tensor
    pixel_query_features: Tensor
    projected_pixel_queries: Tensor
    jpeg_bytes: Tensor
    byte_features: Tensor
    byte_attention_mask: Tensor
    byte_query_features: Tensor
    projected_byte_queries: Tensor
    concatenated_visual_tokens: Tensor
    vicuna_inputs_embeds: Tensor
    vicuna_attention_mask: Tensor
    labels: Tensor
    pixel_query_slice: slice
    byte_query_slice: slice
    prompt_slices: list[slice]
    answer_slices: list[slice]
    condition: str

    @property
    def inputs_embeds(self) -> Tensor:
        return self.vicuna_inputs_embeds

    @property
    def attention_mask(self) -> Tensor:
        return self.vicuna_attention_mask


@dataclass
class DualBranchOutput:
    loss: Tensor
    logits: Tensor
    prepared: DualBranchInputs


def _validate_derangement(indices: Tensor | None, batch_size: int, name: str) -> Tensor:
    if indices is None:
        raise ValueError(f"{name} requires explicit shuffled_indices")
    values = indices.to(dtype=torch.long)
    if values.shape != (batch_size,):
        raise ValueError("shuffled_indices must contain one source index per sample")
    identity = torch.arange(batch_size, device=values.device)
    if batch_size < 3 or not bool(values.ne(identity).all().item()):
        raise ValueError("Dual-branch shuffling requires at least three fixed-point-free samples")
    if sorted(values.cpu().tolist()) != list(range(batch_size)):
        raise ValueError("shuffled_indices must be a permutation")
    return values


class DualBranchVicuna(ByteVicuna):
    """Concatenate Pixel then Byte Query Tokens without learning a fusion rule."""

    def __init__(
        self,
        *,
        visual_encoder: nn.Module,
        image_bridge: nn.Module,
        byteformer_adapter: nn.Module,
        byte_bridge: nn.Module,
        vicuna: nn.Module,
        llm_tokenizer: Any,
        qformer_tokenizer: Any,
        pixel_weight: float = 1.0,
        byte_weight: float = 1.0,
        max_instruction_length: int = 128,
        max_answer_length: int = 256,
    ) -> None:
        if not torch.isfinite(torch.tensor([pixel_weight, byte_weight])).all():
            raise ValueError("Pixel and Byte weights must be finite")
        super().__init__(
            byteformer_adapter=byteformer_adapter,
            bridge=byte_bridge,
            vicuna=vicuna,
            llm_tokenizer=llm_tokenizer,
            qformer_tokenizer=qformer_tokenizer,
            max_instruction_length=max_instruction_length,
            max_answer_length=max_answer_length,
        )
        self.visual_encoder = visual_encoder
        self.image_bridge = image_bridge
        self.pixel_weight = float(pixel_weight)
        self.byte_weight = float(byte_weight)
        if self.image_bridge.llm_projector.out_features != vicuna.config.hidden_size:
            raise ValueError("Image projector output does not equal Vicuna hidden size")
        freeze_byteformer(self.visual_encoder)
        self.image_bridge.requires_grad_(False)
        self.bridge.requires_grad_(False)
        self.vicuna.requires_grad_(False)
        self.eval()

    @property
    def byte_bridge(self) -> nn.Module:
        return self.bridge

    def train(self, mode: bool = True):
        # This integration stage is inference-only.  Ignore attempts to enable
        # stochastic train behavior so every branch remains frozen and stable.
        super(ByteVicuna, self).train(False)
        self.visual_encoder.eval()
        self.image_bridge.eval()
        self.byteformer_adapter.eval()
        self.bridge.eval()
        self.vicuna.eval()
        return self

    def _encode_bridges(
        self,
        pixel_features: Tensor,
        pixel_mask: Tensor,
        byte_features: Tensor,
        byte_mask: Tensor,
        questions: list[str],
    ) -> tuple[ImageQFormerOutput, ByteQFormerBridgeOutput]:
        instruction_ids, instruction_mask = self._qformer_tokens(questions)
        pixel_output = self.image_bridge(
            pixel_features,
            pixel_mask.long(),
            instruction_ids,
            instruction_mask,
        )
        byte_output = self.bridge(
            encoder_hidden_states=byte_features,
            encoder_attention_mask=byte_mask.long(),
            input_ids=instruction_ids,
            text_attention_mask=instruction_mask,
        )
        return pixel_output, byte_output

    def encode_online(
        self,
        pixel_values: Tensor,
        jpeg_bytes: Tensor,
        questions: str | Sequence[str],
    ) -> tuple[ImageQFormerOutput, ByteQFormerBridgeOutput, Tensor, Tensor, Tensor, Tensor]:
        if pixel_values.ndim != 4 or pixel_values.shape[1:] != (3, 224, 224):
            raise ValueError("pixel_values must have shape [B,3,224,224]")
        if jpeg_bytes.ndim != 2 or jpeg_bytes.shape[0] != pixel_values.shape[0]:
            raise ValueError("jpeg_bytes must have shape [B,L] and share the Pixel batch")
        questions_batch = self._normalize_text_batch(
            questions, pixel_values.shape[0], "questions"
        )
        with torch.no_grad():
            pixel_features = self.visual_encoder(pixel_values)
            byte_features, byte_mask = self.byteformer_adapter(jpeg_bytes)
        pixel_mask = torch.ones(
            pixel_features.shape[:2], dtype=torch.long, device=pixel_features.device
        )
        pixel_output, byte_output = self._encode_bridges(
            pixel_features,
            pixel_mask,
            byte_features,
            byte_mask,
            questions_batch,
        )
        return (
            pixel_output,
            byte_output,
            pixel_features,
            pixel_mask,
            byte_features,
            byte_mask,
        )

    def _condition_features(
        self,
        pixel_features: Tensor,
        byte_features: Tensor,
        *,
        condition: str,
        shuffled_indices: Tensor | None,
    ) -> tuple[Tensor, Tensor, float, float]:
        if condition not in DUAL_CONDITIONS:
            raise ValueError(f"Unknown dual condition: {condition}")
        pixel_scale = self.pixel_weight
        byte_scale = self.byte_weight
        if condition == "pixel_only":
            byte_scale = 0.0
        elif condition == "byte_only":
            pixel_scale = 0.0
        elif condition == "zero_both":
            pixel_scale = 0.0
            byte_scale = 0.0
        elif condition == "shuffled_pixel":
            indices = _validate_derangement(
                shuffled_indices, pixel_features.shape[0], condition
            ).to(pixel_features.device)
            pixel_features = pixel_features.index_select(0, indices)
        elif condition == "shuffled_byte":
            indices = _validate_derangement(
                shuffled_indices, byte_features.shape[0], condition
            ).to(byte_features.device)
            byte_features = byte_features.index_select(0, indices)
        return pixel_features, byte_features, pixel_scale, byte_scale

    def prepare_inputs_from_features(
        self,
        pixel_features: Tensor,
        pixel_attention_mask: Tensor,
        byte_features: Tensor,
        byte_attention_mask: Tensor,
        questions: str | Sequence[str],
        answers: str | Sequence[str],
        *,
        condition: str = "pixel_byte",
        shuffled_indices: Tensor | None = None,
        pixel_values: Tensor | None = None,
        jpeg_bytes: Tensor | None = None,
    ) -> DualBranchInputs:
        batch_size = pixel_features.shape[0]
        if pixel_features.ndim != 3 or byte_features.ndim != 3:
            raise ValueError("Pixel and Byte features must both be rank three")
        if byte_features.shape[0] != batch_size:
            raise ValueError("Pixel and Byte feature batches differ")
        if pixel_attention_mask.shape != pixel_features.shape[:2]:
            raise ValueError("Pixel attention mask shape mismatch")
        if byte_attention_mask.shape != byte_features.shape[:2]:
            raise ValueError("Byte attention mask shape mismatch")
        question_batch = self._normalize_text_batch(questions, batch_size, "questions")
        answer_batch = self._normalize_text_batch(answers, batch_size, "answers")
        if any(answer not in {"yes", "no"} for answer in answer_batch):
            raise ValueError("Dual integration only accepts yes/no answers")
        conditioned_pixel, conditioned_byte, pixel_scale, byte_scale = (
            self._condition_features(
                pixel_features,
                byte_features,
                condition=condition,
                shuffled_indices=shuffled_indices,
            )
        )
        pixel_output, byte_output = self._encode_bridges(
            conditioned_pixel,
            pixel_attention_mask,
            conditioned_byte,
            byte_attention_mask,
            question_batch,
        )
        projected_pixel = pixel_output.projected_features * pixel_scale
        projected_byte = byte_output.projected_features * byte_scale
        visual_tokens = torch.cat([projected_pixel, projected_byte], dim=1)

        fixed_ids, fixed_mask = self._fixed_text_prefix(batch_size)
        prompt_ids, prompt_mask = self._token_ids(
            [f"{question.strip()}{XINSTRUCT_POSTFIX}" for question in question_batch],
            add_special_tokens=False,
            truncation_side="left",
            max_length=self.max_instruction_length,
        )
        answer_ids, answer_mask = self._token_ids(
            [f"{answer}{self.llm_tokenizer.eos_token}" for answer in answer_batch],
            add_special_tokens=True,
            truncation_side="right",
            max_length=self.max_answer_length,
        )

        text_rows: list[Tensor] = []
        label_rows: list[Tensor] = []
        prompt_lengths: list[int] = []
        answer_lengths: list[int] = []
        for index in range(batch_size):
            prompt_length = int(prompt_mask[index].sum().item())
            prompt_row = prompt_ids[index, :prompt_length]
            raw_answer_length = int(answer_mask[index].sum().item())
            answer_row = answer_ids[index, 1:raw_answer_length]
            if answer_row.numel() != 2 or int(answer_row[-1]) != int(
                self.llm_tokenizer.eos_token_id
            ):
                raise RuntimeError("Answer-only labels require one yes/no token plus EOS")
            labels = torch.full_like(answer_row, -100)
            labels[0] = answer_row[0]
            text_rows.append(torch.cat([prompt_row, answer_row]))
            label_rows.append(torch.cat([torch.full_like(prompt_row, -100), labels]))
            prompt_lengths.append(prompt_length)
            answer_lengths.append(int(answer_row.numel()))

        text_width = max(int(row.numel()) for row in text_rows)
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
        for index, (row, labels) in enumerate(zip(text_rows, label_rows)):
            text_ids[index, : row.numel()] = row
            text_mask[index, : row.numel()] = 1
            text_labels[index, : row.numel()] = labels

        embedding = self.vicuna.get_input_embeddings()
        dtype = embedding.weight.dtype
        fixed_embeds = embedding(fixed_ids).to(dtype=dtype)
        visual_for_llm = visual_tokens.to(dtype=dtype)
        text_embeds = embedding(text_ids).to(dtype=dtype)
        visual_mask = torch.ones(
            visual_tokens.shape[:2], dtype=torch.long, device=self.device
        )
        inputs_embeds = torch.cat([fixed_embeds, visual_for_llm, text_embeds], dim=1)
        attention_mask = torch.cat([fixed_mask, visual_mask, text_mask], dim=1)
        prefix_labels = torch.full(
            (batch_size, fixed_ids.shape[1] + visual_tokens.shape[1]),
            -100,
            dtype=torch.long,
            device=self.device,
        )
        labels = torch.cat([prefix_labels, text_labels], dim=1)

        pixel_start = fixed_ids.shape[1]
        pixel_slice = slice(pixel_start, pixel_start + projected_pixel.shape[1])
        byte_slice = slice(pixel_slice.stop, pixel_slice.stop + projected_byte.shape[1])
        prompt_slices: list[slice] = []
        answer_slices: list[slice] = []
        for prompt_length, answer_length in zip(prompt_lengths, answer_lengths):
            prompt_slices.append(slice(byte_slice.stop, byte_slice.stop + prompt_length))
            answer_slices.append(
                slice(
                    byte_slice.stop + prompt_length,
                    byte_slice.stop + prompt_length + answer_length,
                )
            )
        if inputs_embeds.shape[:2] != attention_mask.shape or labels.shape != attention_mask.shape:
            raise RuntimeError("Dual Vicuna embeddings, mask, and labels must align")
        ignored = int(labels.eq(-100).sum().item())
        supervised = int(labels.ne(-100).sum().item())
        if ignored + supervised != labels.numel() or supervised != batch_size:
            raise RuntimeError("Dual answer-only label conservation failed")
        if not labels[:, pixel_slice].eq(-100).all() or not labels[:, byte_slice].eq(-100).all():
            raise RuntimeError("Visual Query Tokens must never receive labels")

        pixel_marker = (
            pixel_values
            if pixel_values is not None
            else pixel_features.new_empty((batch_size, 0, 0, 0))
        )
        byte_marker = (
            jpeg_bytes
            if jpeg_bytes is not None
            else torch.empty((batch_size, 0), dtype=torch.int32, device=self.device)
        )
        return DualBranchInputs(
            pixel_values=pixel_marker,
            pixel_vision_features=conditioned_pixel,
            pixel_vision_attention_mask=pixel_attention_mask,
            pixel_query_features=pixel_output.query_features,
            projected_pixel_queries=projected_pixel,
            jpeg_bytes=byte_marker,
            byte_features=conditioned_byte,
            byte_attention_mask=byte_attention_mask,
            byte_query_features=byte_output.query_features,
            projected_byte_queries=projected_byte,
            concatenated_visual_tokens=visual_tokens,
            vicuna_inputs_embeds=inputs_embeds,
            vicuna_attention_mask=attention_mask,
            labels=labels,
            pixel_query_slice=pixel_slice,
            byte_query_slice=byte_slice,
            prompt_slices=prompt_slices,
            answer_slices=answer_slices,
            condition=condition,
        )

    def prepare_inputs(
        self,
        pixel_values: Tensor,
        jpeg_bytes: Tensor,
        questions: str | Sequence[str],
        answers: str | Sequence[str],
        *,
        condition: str = "pixel_byte",
        shuffled_indices: Tensor | None = None,
    ) -> DualBranchInputs:
        encoded = self.encode_online(pixel_values, jpeg_bytes, questions)
        return self.prepare_inputs_from_features(
            encoded[2],
            encoded[3],
            encoded[4],
            encoded[5],
            questions,
            answers,
            condition=condition,
            shuffled_indices=shuffled_indices,
            pixel_values=pixel_values,
            jpeg_bytes=jpeg_bytes,
        )

    def _forward_prepared(self, prepared: DualBranchInputs) -> DualBranchOutput:
        output = self.vicuna(
            inputs_embeds=prepared.vicuna_inputs_embeds,
            attention_mask=prepared.vicuna_attention_mask,
            labels=prepared.labels,
            use_cache=False,
            return_dict=True,
        )
        return DualBranchOutput(output.loss, output.logits, prepared)

    def forward(self, pixel_values: Tensor, jpeg_bytes: Tensor, questions, answers, **kwargs):
        return self._forward_prepared(
            self.prepare_inputs(pixel_values, jpeg_bytes, questions, answers, **kwargs)
        )

    def forward_from_features(
        self,
        pixel_features: Tensor,
        pixel_attention_mask: Tensor,
        byte_features: Tensor,
        byte_attention_mask: Tensor,
        questions,
        answers,
        **kwargs,
    ) -> DualBranchOutput:
        return self._forward_prepared(
            self.prepare_inputs_from_features(
                pixel_features,
                pixel_attention_mask,
                byte_features,
                byte_attention_mask,
                questions,
                answers,
                **kwargs,
            )
        )


def dual_parameter_summary(model: DualBranchVicuna) -> dict[str, dict[str, int]]:
    groups = {
        "visual_encoder": model.visual_encoder,
        "image_bridge": model.image_bridge,
        "byteformer": model.byteformer_adapter,
        "byte_bridge": model.byte_bridge,
        "vicuna": model.vicuna,
    }
    return {
        name: {
            "tensors": len(module.state_dict()),
            "parameters": sum(parameter.numel() for parameter in module.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in module.parameters()
                if parameter.requires_grad
            ),
        }
        for name, module in groups.items()
    }
