"""Fast contracts for the independent Pixel-to-Vicuna integration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch import nn

from spacebyte.answer_only import audit_answer_only_labels
from spacebyte.data import (
    OBJECT_COLOR,
    SUBJECT_COLOR,
    fixed_derangement,
    render_record_to_jpeg_bytes,
    render_record_to_pixel_values,
)
from spacebyte.models import (
    ImageQFormerBridge,
    PixelVicuna,
    build_xinstruct_qformer_tokenizer,
    build_xinstruct_vicuna_tokenizer,
)
from spacebyte.models.pixel_vicuna import ImageQFormerOutput


VICUNA_PATH = Path("/data/students/Zhang_jinsong/spacebyte/models/vicuna-7b-v1.1")
QFORMER_TOKENIZER_PATH = Path(
    "/data/students/Zhang_jinsong/spacebyte/models/bert-base-uncased-tokenizer"
)


class TinyVision(nn.Module):
    num_features = 8

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 8)

    def forward(self, pixels):
        pooled = nn.functional.adaptive_avg_pool2d(pixels, (2, 2))
        return self.proj(pooled.flatten(2).transpose(1, 2))


class TinyImageBridge(nn.Module):
    def __init__(self, hidden_size=16) -> None:
        super().__init__()
        self.vision_layernorm = nn.LayerNorm(8)
        self.query_tokens = nn.Parameter(torch.randn(1, 4, 6))
        self.qformer = nn.ModuleDict(
            {
                "self_attention": nn.Linear(6, 6),
                "cross_key": nn.Linear(8, 6),
                "cross_value": nn.Linear(8, 6),
            }
        )
        self.llm_projector = nn.Linear(6, hidden_size)

    def forward(self, encoder_hidden_states, encoder_attention_mask, input_ids, text_attention_mask):
        normalized = self.vision_layernorm(encoder_hidden_states)
        context = normalized.mean(1)
        queries = self.query_tokens.expand(normalized.shape[0], -1, -1)
        queries = self.qformer["self_attention"](queries)
        queries = queries + self.qformer["cross_key"](context)[:, None]
        queries = queries + self.qformer["cross_value"](context)[:, None]
        query_mask = torch.ones(queries.shape[:2], dtype=torch.long)
        return ImageQFormerOutput(
            queries,
            self.llm_projector(queries),
            query_mask,
            torch.cat([query_mask, text_attention_mask], dim=1),
        )


class TinyVicuna(nn.Module):
    def __init__(self, vocab_size: int, hidden_size=16) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, 64, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, labels=None, **_kwargs):
        masked = inputs_embeds * attention_mask.unsqueeze(-1)
        context = masked.cumsum(1) / attention_mask.cumsum(1).clamp_min(1).unsqueeze(-1)
        logits = self.head(context)
        labels_for_loss = labels.clone()
        supervised = labels_for_loss.ne(-100)
        labels_for_loss[supervised] %= logits.shape[-1]
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels_for_loss[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(logits=logits, loss=loss)


def tiny_model() -> PixelVicuna:
    llm_tokenizer = build_xinstruct_vicuna_tokenizer(VICUNA_PATH)
    qformer_tokenizer = build_xinstruct_qformer_tokenizer(QFORMER_TOKENIZER_PATH)
    return PixelVicuna(
        visual_encoder=TinyVision(),
        bridge=TinyImageBridge(),
        vicuna=TinyVicuna(len(llm_tokenizer)),
        llm_tokenizer=llm_tokenizer,
        qformer_tokenizer=qformer_tokenizer,
    )


def test_pixel_uses_exact_shared_marked_jpeg(tmp_path: Path) -> None:
    (tmp_path / "images" / "flickr").mkdir(parents=True)
    image_path = tmp_path / "images" / "flickr" / "sample.jpg"
    Image.new("RGB", (80, 40), (20, 180, 70)).save(image_path)
    record = {
        "sample_id": "same-sample",
        "image_path": "images/flickr/sample.jpg",
        "annotation_width": 80,
        "annotation_height": 40,
        "subject": {"bbox": [5, 25, 5, 25]},
        "object": {"bbox": [10, 35, 45, 75]},
    }
    byte_stream, rendered, boxes = render_record_to_jpeg_bytes(tmp_path, record)
    pixel = render_record_to_pixel_values(tmp_path, record)
    assert torch.equal(pixel.jpeg_bytes, byte_stream)
    assert pixel.sample_id == record["sample_id"]
    assert pixel.boxes == boxes
    assert pixel.pixel_values.shape == (3, 224, 224)
    assert rendered.getpixel(tuple(boxes["subject_xyxy"][:2])) == SUBJECT_COLOR
    assert rendered.getpixel(tuple(boxes["object_xyxy"][:2])) == OBJECT_COLOR


def test_fixed_derangement_is_global_deterministic_and_has_no_identity() -> None:
    first = fixed_derangement(9, 20261014)
    second = fixed_derangement(9, 20261014)
    assert torch.equal(first, second)
    assert first.ne(torch.arange(9)).all()
    assert sorted(first.tolist()) == list(range(9))


def test_answer_only_shapes_ablation_and_gradients() -> None:
    model = tiny_model().train()
    pixels = torch.stack(
        [torch.zeros(3, 224, 224), torch.ones(3, 224, 224), torch.full((3, 224, 224), 2.0)]
    )
    questions = ["Where?", "Where?", "Where?"]
    answers = ["yes", "no", "yes"]
    normal = model(pixels, questions, answers, answer_only_loss=True)
    audit = audit_answer_only_labels(normal.prepared, answers, model.llm_tokenizer)
    assert audit["supervised_token_count"] == 3
    assert audit["eos_fully_ignored"]
    assert (
        audit["ignored_token_count"] + audit["supervised_token_count"]
        == normal.prepared.labels.numel()
    )
    assert normal.prepared.pixel_values.shape == (3, 3, 224, 224)
    assert normal.prepared.vision_features.shape == (3, 4, 8)
    assert normal.prepared.vision_attention_mask.shape == (3, 4)
    assert normal.prepared.image_query_features.shape == (3, 4, 6)
    assert normal.prepared.projected_image_queries.shape == (3, 4, 16)
    assert normal.prepared.vicuna_inputs_embeds.shape[:2] == normal.prepared.labels.shape
    assert normal.prepared.vicuna_attention_mask.shape == normal.prepared.labels.shape
    assert torch.isfinite(normal.loss)

    with torch.no_grad():
        zero = model(pixels, questions, answers, answer_only_loss=True, pixel_condition="zero")
        shuffled = model(
            pixels,
            questions,
            answers,
            answer_only_loss=True,
            pixel_condition="shuffled",
            shuffled_indices=fixed_derangement(3, 7),
        )
    assert not torch.equal(normal.logits, zero.logits)
    assert not torch.equal(normal.logits, shuffled.logits)

    normal.loss.backward()
    assert model.bridge.query_tokens.grad.abs().sum() > 0
    assert model.bridge.qformer["self_attention"].weight.grad.abs().sum() > 0
    assert model.bridge.qformer["cross_key"].weight.grad.abs().sum() > 0
    assert model.bridge.qformer["cross_value"].weight.grad.abs().sum() > 0
    assert model.bridge.llm_projector.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.visual_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.vicuna.parameters())


def test_trainable_checkpoint_roundtrip_is_strict(tmp_path: Path) -> None:
    source = TinyImageBridge()
    path = tmp_path / "pixel_trainable.pt"
    torch.save(source.state_dict(), path)
    target = TinyImageBridge()
    result = target.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    for key, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[key])


def test_official_image_bridge_dimensions() -> None:
    bridge = ImageQFormerBridge()
    assert bridge.vision_layernorm.normalized_shape == (1408,)
    assert bridge.query_tokens.shape == (1, 32, 768)
    assert bridge.llm_projector.in_features == 768
    assert bridge.llm_projector.out_features == 4096
    cross_layers = [
        index
        for index, layer in enumerate(bridge.qformer.bert.encoder.layer)
        if hasattr(layer, "crossattention")
    ]
    assert cross_layers == [0, 2, 4, 6, 8, 10]
