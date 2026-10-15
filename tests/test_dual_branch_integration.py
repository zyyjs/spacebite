"""Fast contracts for frozen Pixel+Byte integration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from spacebyte.answer_only import audit_answer_only_labels
from spacebyte.data import fixed_derangement, render_record_to_pixel_values
from spacebyte.models import (
    ByteQFormerBridgeOutput,
    DualBranchVicuna,
    build_xinstruct_qformer_tokenizer,
    build_xinstruct_vicuna_tokenizer,
)
from spacebyte.models.pixel_vicuna import ImageQFormerOutput
from scripts.run_dual_branch_integration_autonomous import (
    select_verification_records,
    strict_task_bridge_load,
)


VICUNA_PATH = Path("/data/students/Zhang_jinsong/spacebyte/models/vicuna-7b-v1.1")
QFORMER_TOKENIZER_PATH = Path(
    "/data/students/Zhang_jinsong/spacebyte/models/bert-base-uncased-tokenizer"
)


class TinyVision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 8)

    def forward(self, pixels):
        pooled = nn.functional.adaptive_avg_pool2d(pixels, (2, 2))
        return self.projection(pooled.flatten(2).transpose(1, 2))


class TinyByteAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, values):
        mask = values.ne(-1).long()
        features = values.clamp_min(0).float().unsqueeze(-1).repeat(1, 1, 7)
        return features / 255.0, mask


class TinyImageBridge(nn.Module):
    def __init__(self, hidden=16) -> None:
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, 4, 6))
        self.llm_projector = nn.Linear(6, hidden)
        self.cross = nn.Linear(8, 6)

    def forward(self, encoder_hidden_states, encoder_attention_mask, input_ids, text_attention_mask):
        context = encoder_hidden_states.mean(1)
        query = self.query_tokens.expand(context.shape[0], -1, -1) + self.cross(context)[:, None]
        mask = torch.ones(query.shape[:2], dtype=torch.long, device=query.device)
        return ImageQFormerOutput(
            query, self.llm_projector(query), mask, torch.cat([mask, text_attention_mask], 1)
        )


class TinyByteBridge(nn.Module):
    def __init__(self, hidden=16) -> None:
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, 4, 6))
        self.llm_projector = nn.Linear(6, hidden)
        self.cross = nn.Linear(7, 6)

    def forward(self, encoder_hidden_states, encoder_attention_mask, input_ids, text_attention_mask):
        weighted = encoder_hidden_states * encoder_attention_mask.unsqueeze(-1)
        context = weighted.sum(1) / encoder_attention_mask.sum(1).clamp_min(1).unsqueeze(-1)
        query = self.query_tokens.expand(context.shape[0], -1, -1) + self.cross(context)[:, None]
        mask = torch.ones(query.shape[:2], dtype=torch.long, device=query.device)
        return ByteQFormerBridgeOutput(
            query, self.llm_projector(query), mask, torch.cat([mask, text_attention_mask], 1)
        )


class TinyVicuna(nn.Module):
    def __init__(self, vocab_size: int, hidden=16) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, 64, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, labels=None, **_kwargs):
        context = (inputs_embeds * attention_mask.unsqueeze(-1)).cumsum(1)
        denominator = attention_mask.cumsum(1).clamp_min(1).unsqueeze(-1)
        logits = self.head(context / denominator)
        targets = labels.clone()
        targets[targets.ne(-100)] %= logits.shape[-1]
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            targets[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(logits=logits, loss=loss)


@pytest.fixture()
def model():
    if not VICUNA_PATH.joinpath("tokenizer.model").is_file():
        pytest.skip("Vicuna tokenizer is unavailable")
    llm = build_xinstruct_vicuna_tokenizer(VICUNA_PATH)
    qformer = build_xinstruct_qformer_tokenizer(QFORMER_TOKENIZER_PATH)
    return DualBranchVicuna(
        visual_encoder=TinyVision(),
        image_bridge=TinyImageBridge(),
        byteformer_adapter=TinyByteAdapter(),
        byte_bridge=TinyByteBridge(),
        vicuna=TinyVicuna(len(llm)),
        llm_tokenizer=llm,
        qformer_tokenizer=qformer,
    )


def inputs():
    pixels = torch.stack(
        [torch.zeros(3, 224, 224), torch.ones(3, 224, 224), torch.full((3, 224, 224), 2.0)]
    )
    jpeg = torch.tensor(
        [[255, 216, 1, -1, -1], [255, 216, 2, 3, -1], [255, 216, 4, 5, 6]],
        dtype=torch.int32,
    )
    return pixels, jpeg, ["Where?"] * 3, ["yes", "no", "yes"]


def test_dual_shapes_order_masks_and_answer_only(model) -> None:
    pixels, jpeg, questions, answers = inputs()
    output = model(pixels, jpeg, questions, answers)
    prepared = output.prepared
    assert prepared.projected_pixel_queries.shape == (3, 4, 16)
    assert prepared.projected_byte_queries.shape == (3, 4, 16)
    assert prepared.concatenated_visual_tokens.shape == (3, 8, 16)
    assert torch.equal(
        prepared.concatenated_visual_tokens[:, :4], prepared.projected_pixel_queries
    )
    assert torch.equal(
        prepared.concatenated_visual_tokens[:, 4:], prepared.projected_byte_queries
    )
    assert prepared.pixel_query_slice.stop == prepared.byte_query_slice.start
    assert prepared.inputs_embeds.shape[:2] == prepared.attention_mask.shape
    assert prepared.labels.shape == prepared.attention_mask.shape
    audit = audit_answer_only_labels(prepared, answers, model.llm_tokenizer)
    assert audit["supervised_token_count"] == 3
    assert audit["eos_fully_ignored"]
    assert audit["ignored_token_count"] + audit["supervised_token_count"] == prepared.labels.numel()
    assert torch.isfinite(output.loss) and torch.isfinite(output.logits).all()


def test_fixed_weights_and_all_six_conditions(model) -> None:
    pixels, jpeg, questions, answers = inputs()
    permutation = fixed_derangement(3, 7)
    with torch.no_grad():
        normal = model(pixels, jpeg, questions, answers)
        pixel_only = model(pixels, jpeg, questions, answers, condition="pixel_only")
        byte_only = model(pixels, jpeg, questions, answers, condition="byte_only")
        zero = model(pixels, jpeg, questions, answers, condition="zero_both")
        shuffled_pixel = model(
            pixels, jpeg, questions, answers, condition="shuffled_pixel", shuffled_indices=permutation
        )
        shuffled_byte = model(
            pixels, jpeg, questions, answers, condition="shuffled_byte", shuffled_indices=permutation
        )
    assert model.pixel_weight == 1.0 and model.byte_weight == 1.0
    assert pixel_only.prepared.projected_byte_queries.eq(0).all()
    assert torch.equal(pixel_only.prepared.projected_pixel_queries, normal.prepared.projected_pixel_queries)
    assert byte_only.prepared.projected_pixel_queries.eq(0).all()
    assert torch.equal(byte_only.prepared.projected_byte_queries, normal.prepared.projected_byte_queries)
    assert zero.prepared.concatenated_visual_tokens.eq(0).all()
    for changed in (pixel_only, byte_only, zero, shuffled_pixel, shuffled_byte):
        assert not torch.equal(changed.logits, normal.logits)


def test_cached_path_matches_online_query_outputs_and_is_deterministic(model) -> None:
    pixels, jpeg, questions, answers = inputs()
    with torch.no_grad():
        encoded = model.encode_online(pixels, jpeg, questions)
        first = model.forward_from_features(
            encoded[2], encoded[3], encoded[4], encoded[5], questions, answers
        )
        second = model.forward_from_features(
            encoded[2], encoded[3], encoded[4], encoded[5], questions, answers
        )
    assert torch.equal(first.prepared.projected_pixel_queries, encoded[0].projected_features)
    assert torch.equal(first.prepared.projected_byte_queries, encoded[1].projected_features)
    assert torch.equal(first.logits, second.logits)


def test_every_parameter_is_frozen_and_unchanged(model) -> None:
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name in {
            "image_bridge.query_tokens",
            "bridge.query_tokens",
            "visual_encoder.projection.weight",
            "byteformer_adapter.scale",
            "vicuna.embedding.weight",
        }
    }
    pixels, jpeg, questions, answers = inputs()
    with torch.no_grad():
        model(pixels, jpeg, questions, answers)
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    model.train(True)
    assert not model.training
    assert not model.visual_encoder.training
    assert not model.image_bridge.training
    assert not model.byteformer_adapter.training
    assert not model.byte_bridge.training
    assert not model.vicuna.training


def test_pixel_and_byte_share_exact_rendered_jpeg(tmp_path: Path) -> None:
    (tmp_path / "images" / "flickr").mkdir(parents=True)
    Image.new("RGB", (90, 50), (20, 80, 160)).save(
        tmp_path / "images" / "flickr" / "same.jpg"
    )
    record = {
        "sample_id": "shared",
        "image_path": "images/flickr/same.jpg",
        "annotation_width": 90,
        "annotation_height": 50,
        "subject": {"bbox": [2, 20, 3, 30]},
        "object": {"bbox": [15, 45, 50, 85]},
    }
    shared = render_record_to_pixel_values(tmp_path, record)
    # Pixel values are decoded from this exact tensor; the Byte branch receives
    # the very same object rather than independently re-rendered content.
    assert shared.jpeg_bytes.dtype == torch.int32
    assert shared.jpeg_bytes.ndim == 1
    assert shared.pixel_values.shape == (3, 224, 224)
    assert shared.sample_id == "shared"


def test_shuffle_rejects_batch_one_or_self_mapping(model) -> None:
    pixels, jpeg, questions, answers = inputs()
    with pytest.raises(ValueError, match="at least three"):
        model(
            pixels[:1], jpeg[:1], questions[:1], answers[:1],
            condition="shuffled_pixel", shuffled_indices=torch.tensor([0]),
        )
    with pytest.raises(ValueError, match="fixed-point-free"):
        model(
            pixels, jpeg, questions, answers,
            condition="shuffled_byte", shuffled_indices=torch.arange(3),
        )


def test_record_selection_forbids_test_and_keeps_train_valid_only() -> None:
    records = [
        {"sample_id": "a", "split": "train", "image_path": "a.jpg", "relation_raw": "on"},
        {"sample_id": "b", "split": "valid", "image_path": "b.jpg", "relation_raw": "above"},
        {"sample_id": "c", "split": "train", "image_path": "c.jpg", "relation_raw": "under"},
    ]
    selected = select_verification_records(records, 3)
    assert {row["split"] for row in selected} == {"train", "valid"}
    with pytest.raises(RuntimeError, match="forbidden split"):
        select_verification_records([*records, {**records[0], "split": "test"}], 3)


def test_task_checkpoint_loader_is_strict_and_exact(tmp_path: Path) -> None:
    source = TinyImageBridge()
    checkpoint = tmp_path / "task.pt"
    torch.save({"bridge_state_dict": source.state_dict()}, checkpoint)
    target = TinyImageBridge()
    report = strict_task_bridge_load(target, checkpoint, "bridge_state_dict")
    assert report["missing_keys"] == []
    assert report["unexpected_keys"] == []
    assert report["shape_mismatch"] == {}
    assert report["post_load_all_tensors_exact"]
    assert all(
        torch.equal(value, target.state_dict()[name])
        for name, value in source.state_dict().items()
    )
