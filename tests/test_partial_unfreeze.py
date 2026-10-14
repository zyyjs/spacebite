"""Contracts for online bytes and identity-based partial ByteFormer unfreezing."""

from __future__ import annotations

import json

import torch
from PIL import Image
from torch import nn

from spacebyte.evaluation import binary_classification_metrics
from spacebyte.partial_unfreeze import (
    SpatialSenseOnlineByteDataset,
    build_partial_unfreeze_optimizer_groups,
    collate_online_bytes,
    configure_byteformer_partial_unfreeze,
)


def make_dataset(root) -> None:
    image_dir = root / "images" / "flickr"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (100, 50), "white").save(image_dir / "sample.jpg")
    annotations = []
    for index, label in enumerate((True, False)):
        annotations.append(
            {
                "_id": f"id-{index}",
                "predicate": "above",
                "subject": {"name": "subject", "bbox": [5, 30, 5, 30]},
                "object": {"name": "object", "bbox": [5, 30, 60, 90]},
                "label": label,
            }
        )
    (root / "annotations.json").write_text(
        json.dumps(
            [
                {
                    "url": "https://example.test/sample.jpg",
                    "height": 50,
                    "width": 100,
                    "annotations": annotations,
                    "split": "train",
                }
            ]
        ),
        encoding="utf-8",
    )


class FakeByteFormer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(257, 8)
        self.token_reduction_net = nn.Conv1d(8, 8, 3)
        self.downsamplers = nn.ModuleDict({"merge": nn.Linear(8, 8)})
        self.transformer = nn.ModuleList([nn.Linear(8, 8) for _ in range(12)])
        self.post_transformer_norm = nn.LayerNorm(8)
        self.classifier = nn.Linear(8, 1000)


class FakeAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.byteformer = FakeByteFormer()


class FakeBridge(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, 2, 8))
        self.qformer = nn.Linear(8, 8)
        self.byte_layernorm = nn.LayerNorm(8)
        self.llm_projector = nn.Linear(8, 16)


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.byteformer_adapter = FakeAdapter()
        self.bridge = FakeBridge()
        self.vicuna = nn.Linear(16, 16).requires_grad_(False)


def test_identity_based_unfreeze_selects_last_two_blocks_and_norm() -> None:
    model = FakeModel()
    report = configure_byteformer_partial_unfreeze(model)
    assert report["transformer_block_count"] == 12
    assert report["unfrozen_modules"] == [
        "transformer.10",
        "transformer.11",
        "post_transformer_norm",
    ]
    trainable = {
        name
        for name, parameter in model.byteformer_adapter.byteformer.named_parameters()
        if parameter.requires_grad
    }
    assert trainable
    assert all(
        name.startswith(
            ("transformer.10.", "transformer.11.", "post_transformer_norm.")
        )
        for name in trainable
    )
    assert not any(
        parameter.requires_grad
        for parameter in model.byteformer_adapter.byteformer.classifier.parameters()
    )

    groups, optimizer_report = build_partial_unfreeze_optimizer_groups(model)
    assert [group["lr"] for group in groups] == [1e-6, 5e-6, 5e-5]
    assert optimizer_report["all_trainable_parameters_assigned_once"]
    assigned = [id(parameter) for group in groups for parameter in group["params"]]
    expected = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    assert sorted(assigned) == sorted(expected)


def test_online_dataset_renders_jpeg_bytes_without_feature_cache(tmp_path) -> None:
    make_dataset(tmp_path)
    from spacebyte.data import flatten_spatialsense_annotations

    records = flatten_spatialsense_annotations(tmp_path, include_splits={"train"})
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "visual_cache_key": f"visual-{index}",
                    "question": row["question"],
                    "answer": row["answer"],
                    "relation": row["relation"],
                    "relation_raw": row["relation_raw"],
                    "label": row["label"],
                }
            )
            + "\n"
            for index, row in enumerate(records)
        ),
        encoding="utf-8",
    )
    dataset = SpatialSenseOnlineByteDataset(
        tmp_path, manifest, expected_split="train"
    )
    first, second = dataset[0], dataset[1]
    assert first["byte_tensor"].dtype == torch.int32
    assert first["byte_tensor"][:2].tolist() == [255, 216]
    batch = collate_online_bytes([first, second])
    assert batch["byte_tensor"].shape[0] == 2
    assert batch["byte_tensor"].dtype == torch.int32
    assert "byte_features" not in batch


def test_binary_metrics_report_macro_f1() -> None:
    metrics = binary_classification_metrics(
        [True, False], [False, False], [-1.0, -1.0]
    )
    assert metrics["accuracy"] == 0.5
    assert metrics["f1"] == 0.0
    assert metrics["macro_f1"] == 1 / 3
