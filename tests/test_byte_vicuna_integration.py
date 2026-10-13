"""Fast contract tests for the Byte Q-Former/Vicuna integration wrapper."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from spacebyte.models import (
    ByteQFormerBridgeOutput,
    ByteVicuna,
    build_xinstruct_qformer_tokenizer,
    build_xinstruct_vicuna_tokenizer,
)
from spacebyte.training import audit_causal_labels


VICUNA_PATH = Path(
    "/data/students/Zhang_jinsong/spacebyte/models/vicuna-7b-v1.1"
)
QFORMER_TOKENIZER_PATH = Path(
    "/data/students/Zhang_jinsong/spacebyte/models/bert-base-uncased-tokenizer"
)


class TinyByteAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen_parameter = nn.Parameter(torch.ones(1))

    def forward(self, byte_tensor):
        mask = byte_tensor.ne(-1).long()
        values = byte_tensor.clamp_min(0).float().unsqueeze(-1).repeat(1, 1, 192)
        return values / 255.0, mask


class TinyBridge(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.llm_projector = nn.Linear(8, hidden_size)
        self.query_tokens = nn.Parameter(torch.randn(1, 32, 8))

    def forward(
        self,
        encoder_hidden_states,
        encoder_attention_mask,
        input_ids,
        text_attention_mask,
    ):
        masked = encoder_hidden_states * encoder_attention_mask.unsqueeze(-1)
        byte_scalar = masked.sum((1, 2)) / encoder_attention_mask.sum(1).clamp_min(1)
        query = self.query_tokens.expand(encoder_hidden_states.shape[0], -1, -1)
        query = query + byte_scalar[:, None, None] / 1000.0
        projected = self.llm_projector(query)
        query_mask = torch.ones(query.shape[:2], dtype=torch.long)
        return ByteQFormerBridgeOutput(
            query_features=query,
            projected_features=projected,
            query_attention_mask=query_mask,
            combined_attention_mask=torch.cat([query_mask, text_attention_mask], 1),
        )


class TinyVicuna(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, 64, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, labels=None, **kwargs):
        # A tiny causal mixer is enough to prove that answer loss reaches the
        # earlier query prefix without allocating a 32k-wide test head.
        masked = inputs_embeds * attention_mask.unsqueeze(-1)
        denominator = attention_mask.cumsum(dim=1).clamp_min(1).unsqueeze(-1)
        causal_context = masked.cumsum(dim=1) / denominator
        logits = self.head(causal_context)
        loss = None
        if labels is not None:
            tiny_labels = labels.clone()
            supervised = tiny_labels.ne(-100)
            tiny_labels[supervised] = tiny_labels[supervised].remainder(logits.shape[-1])
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                tiny_labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(logits=logits, loss=loss)

    def generate(self, inputs_embeds, attention_mask, **kwargs):
        return torch.tensor([[2, 2175, 310]], device=inputs_embeds.device)


@pytest.fixture(scope="module")
def tiny_model():
    if not (VICUNA_PATH.joinpath("tokenizer.model").exists() and QFORMER_TOKENIZER_PATH.exists()):
        pytest.skip("Local official tokenizer assets are unavailable")
    llm_tokenizer = build_xinstruct_vicuna_tokenizer(VICUNA_PATH)
    qformer_tokenizer = build_xinstruct_qformer_tokenizer(QFORMER_TOKENIZER_PATH)
    model = ByteVicuna(
        byteformer_adapter=TinyByteAdapter(),
        bridge=TinyBridge(hidden_size=16),
        vicuna=TinyVicuna(len(llm_tokenizer)),
        llm_tokenizer=llm_tokenizer,
        qformer_tokenizer=qformer_tokenizer,
    )
    return model


def sample_bytes():
    return torch.tensor(
        [[255, 216, 1, 2, 3, -1, -1], [255, 216, 4, 5, 6, 7, 8]],
        dtype=torch.int32,
    )


def test_xinstruct_tokenizer_special_tokens(tiny_model) -> None:
    tokenizer = tiny_model.llm_tokenizer
    assert len(tokenizer) == 32001
    assert tokenizer.bos_token_id == 2
    assert tokenizer.eos_token_id == 2
    assert tokenizer.unk_token_id == 2
    assert tokenizer.pad_token == "[PAD]"
    assert tokenizer.pad_token_id == 32000
    assert len(tiny_model.qformer_tokenizer) == 30523
    assert tiny_model.qformer_tokenizer.bos_token == "[DEC]"


def test_prompt_query_and_label_shapes(tiny_model) -> None:
    tiny_model.eval()
    output = tiny_model(
        sample_bytes(),
        ["Where is A?", "Where is the subject?"],
        ["left of", "above"],
    )
    prepared = output.prepared
    assert prepared.projected_queries.shape == (2, 32, 16)
    assert prepared.inputs_embeds.shape[:2] == prepared.attention_mask.shape
    assert prepared.labels.shape == prepared.attention_mask.shape
    assert prepared.query_slice.start == 8
    assert prepared.query_slice.stop - prepared.query_slice.start == 32
    assert prepared.labels[:, : prepared.query_slice.start].eq(-100).all()
    assert prepared.labels[:, prepared.query_slice].eq(-100).all()
    for index, prompt_slice in enumerate(prepared.prompt_slices):
        assert prepared.labels[index, prompt_slice].eq(-100).all()
    for index, answer_slice in enumerate(prepared.answer_slices):
        assert prepared.labels[index, answer_slice].ne(-100).all()
    assert prepared.labels[prepared.attention_mask.eq(0)].eq(-100).all()
    audit = audit_causal_labels(
        prepared, ["left of", "above"], tiny_model.llm_tokenizer
    )
    assert audit["ignored"] + audit["supervised"] == audit["numel"]
    assert audit["shape"] == list(prepared.attention_mask.shape)
    assert [row["answer"] for row in audit["per_sample"]] == ["left of", "above"]
    assert torch.isfinite(output.loss)


def test_cached_feature_forward_is_finite_and_differentiable(tiny_model) -> None:
    tiny_model.train()
    tiny_model.zero_grad(set_to_none=True)
    with torch.no_grad():
        features, mask = tiny_model.byteformer_adapter(sample_bytes()[:1])
    output = tiny_model.forward_from_byte_features(features, mask, "Where?", "yes")
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert tiny_model.bridge.query_tokens.grad is not None
    assert tiny_model.bridge.query_tokens.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in tiny_model.vicuna.parameters())
    assert all(
        parameter.grad is None for parameter in tiny_model.byteformer_adapter.parameters()
    )


def test_frozen_modules_and_trainable_bridge(tiny_model) -> None:
    tiny_model.train()
    assert tiny_model.bridge.training
    assert not tiny_model.byteformer_adapter.training
    assert not tiny_model.vicuna.training
    assert not any(p.requires_grad for p in tiny_model.byteformer_adapter.parameters())
    assert not any(p.requires_grad for p in tiny_model.vicuna.parameters())
    assert all(p.requires_grad for p in tiny_model.bridge.parameters())


def test_text_padding_does_not_change_valid_logits(tiny_model) -> None:
    tiny_model.eval()
    questions = ["Where?", "Where is the subject relative to the object?"]
    answers = ["left", "above and behind"]
    with torch.no_grad():
        baseline = tiny_model(sample_bytes(), questions, answers)
        padded = tiny_model(
            sample_bytes(),
            questions,
            answers,
            extra_text_padding=5,
        )
    valid = baseline.prepared.attention_mask.bool()
    padded_prefix = padded.logits[:, : baseline.logits.shape[1]]
    assert torch.allclose(
        baseline.logits[valid],
        padded_prefix[valid],
        atol=1e-6,
        rtol=0,
    )


def test_query_prefix_changes_logits_and_carries_gradient(tiny_model) -> None:
    tiny_model.eval()
    tiny_model.zero_grad(set_to_none=True)
    baseline = tiny_model(sample_bytes()[:1], "Where?", "left")
    zeroed = tiny_model(
        sample_bytes()[:1], "Where?", "left", projected_query_scale=0.0
    )
    assert not torch.equal(baseline.logits, zeroed.logits)
    baseline.loss.backward()
    assert tiny_model.bridge.query_tokens.grad is not None
    assert tiny_model.bridge.query_tokens.grad.abs().sum() > 0
    assert tiny_model.bridge.llm_projector.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in tiny_model.vicuna.parameters())
    assert all(p.grad is None for p in tiny_model.byteformer_adapter.parameters())


def test_generation_returns_decodable_tokens(tiny_model) -> None:
    text, ids, prepared = tiny_model.generate(sample_bytes()[:1], "Where?")
    assert ids.numel() > 0
    assert isinstance(text[0], str)
    assert prepared.projected_queries.shape == (1, 32, 16)

    left_padded = tiny_model.prepare_inputs(
        sample_bytes(),
        ["Where?", "Where is the subject relative to the object?"],
        text_padding_side="left",
    )
    text_start = left_padded.query_slice.stop
    assert left_padded.attention_mask[0, text_start] == 0
    assert left_padded.attention_mask[0, -1] == 1
