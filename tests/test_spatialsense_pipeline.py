"""Contracts for the SpatialSense binary instruction and cache pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from spacebyte.data import (
    RAW_RELATIONS,
    SpatialSenseCachedFeatureDataset,
    build_tiny_overfit_manifest,
    collate_cached_features,
    encode_marked_jpeg,
    flatten_spatialsense_annotations,
    normalize_relation,
    parse_binary_answer,
    render_marked_image,
)
from spacebyte.training import (
    assert_frozen_modules_absent,
    build_bridge_optimizer_groups,
)


def make_dataset(root: Path) -> None:
    image_dir = root / "images" / "flickr"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (100, 50), "white").save(image_dir / "sample.jpg")
    annotations = []
    number = 0
    for relation in RAW_RELATIONS:
        for label in (False, True):
            for duplicate in range(2):
                annotations.append(
                    {
                        "_id": f"id-{number}",
                        "predicate": relation,
                        "subject": {
                            "name": "subject",
                            "bbox": [5, 30, 5, 30],
                            "x": 10,
                            "y": 10,
                        },
                        "object": {
                            "name": "object",
                            "bbox": [5, 30, 60, 90],
                            "x": 70,
                            "y": 10,
                        },
                        "label": label,
                    }
                )
                number += 1
    image_record = {
        "url": "https://example.test/sample.jpg",
        "height": 50,
        "width": 100,
        "annotations": annotations,
        "split": "train",
    }
    (root / "annotations.json").write_text(
        json.dumps([image_record]), encoding="utf-8"
    )


def test_data_field_parsing_and_binary_semantics(tmp_path) -> None:
    make_dataset(tmp_path)
    records = flatten_spatialsense_annotations(tmp_path)
    assert len(records) == 36
    assert {row["answer"] for row in records} == {"yes", "no"}
    assert all(row["split"] == "train" for row in records)
    assert all(row["subject"]["bbox"] == [5.0, 30.0, 5.0, 30.0] for row in records)
    assert records[0]["question"].endswith("Answer yes or no.")


def test_relation_normalization_and_fixed_manifest(tmp_path) -> None:
    make_dataset(tmp_path)
    records = flatten_spatialsense_annotations(tmp_path)
    assert normalize_relation("to the left of") == "left of"
    assert normalize_relation("to the right of") == "right of"
    manifest = build_tiny_overfit_manifest(records, seed=7)
    assert manifest["sample_count"] == 36
    ids = [row["sample_id"] for row in manifest["samples"]]
    assert len(ids) == len(set(ids))
    assert all(counts == {"no": 2, "yes": 2} for counts in manifest["class_counts"].values())


def test_subject_red_object_blue_and_jpeg_q100() -> None:
    image = Image.new("RGB", (100, 50), "white")
    rendered, boxes = render_marked_image(
        image,
        subject_bbox=[5, 30, 5, 30],
        object_bbox=[5, 30, 60, 90],
        annotation_width=100,
        annotation_height=50,
    )
    subject_x, subject_y = boxes["subject_xyxy"][:2]
    object_x, object_y = boxes["object_xyxy"][:2]
    assert rendered.getpixel((subject_x, subject_y)) == (255, 32, 32)
    assert rendered.getpixel((object_x, object_y)) == (0, 96, 255)
    first = encode_marked_jpeg(rendered)
    second = encode_marked_jpeg(rendered)
    assert torch.equal(first, second)
    assert first[:2].tolist() == [255, 216]
    assert first[-2:].tolist() == [255, 217]


def test_cached_dataset_padding_and_fp16_consistency(tmp_path) -> None:
    records = [
        {
            "sample_id": "a",
            "question": "q1",
            "answer": "yes",
            "relation": "above",
            "relation_raw": "above",
            "label": True,
        },
        {
            "sample_id": "b",
            "question": "q2",
            "answer": "no",
            "relation": "under",
            "relation_raw": "under",
            "label": False,
        },
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": records}), encoding="utf-8")
    original = {}
    for sample_id, length in (("a", 3), ("b", 5)):
        features = torch.linspace(-1, 1, length * 192).reshape(length, 192)
        original[sample_id] = features
        torch.save(
            {
                "sample_id": sample_id,
                "byte_features": features.half(),
                "byte_attention_mask": torch.ones(length, dtype=torch.uint8),
            },
            tmp_path / f"{sample_id}.pt",
        )
    dataset = SpatialSenseCachedFeatureDataset(manifest, tmp_path)
    batch = collate_cached_features([dataset[0], dataset[1]])
    assert batch["byte_features"].shape == (2, 5, 192)
    assert batch["byte_attention_mask"].tolist() == [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
    ]
    error = (dataset[0]["byte_features"].float() - original["a"]).abs().max()
    assert error <= 5e-4


class OptimizerBridge(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, 2, 4))
        self.byte_layernorm = nn.LayerNorm(4)
        self.llm_projector = nn.Linear(4, 4)
        self.qformer = nn.Module()
        self.qformer.bert = nn.Module()
        self.qformer.bert.encoder = nn.Module()
        self.qformer.bert.encoder.layer = nn.ModuleList()
        for index in range(12):
            layer = nn.Module()
            layer.attention = nn.Linear(4, 4)
            if index % 2 == 0:
                layer.crossattention = nn.Module()
                setattr(layer.crossattention, "self", nn.Module())
                layer.crossattention.self.key = nn.Linear(4, 4)
                layer.crossattention.self.value = nn.Linear(4, 4)
            self.qformer.bert.encoder.layer.append(layer)


def test_optimizer_groups_are_exhaustive_disjoint_and_update_only_bridge() -> None:
    bridge = OptimizerBridge()
    frozen_byteformer = nn.Linear(4, 4).requires_grad_(False)
    frozen_vicuna = nn.Linear(4, 4).requires_grad_(False)
    groups, report = build_bridge_optimizer_groups(bridge)
    assert report["all_trainable_parameters_assigned_once"]
    assigned = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == {
        id(parameter) for parameter in bridge.parameters() if parameter.requires_grad
    }
    assert_frozen_modules_absent(groups, frozen_byteformer, frozen_vicuna)
    before_bridge = {name: value.detach().clone() for name, value in bridge.named_parameters()}
    before_frozen = [value.detach().clone() for value in frozen_byteformer.parameters()]
    optimizer = torch.optim.AdamW(groups)
    loss = sum(parameter.square().sum() for parameter in bridge.parameters())
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert any(
        not torch.equal(before_bridge[name], value)
        for name, value in bridge.named_parameters()
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(before_frozen, frozen_byteformer.parameters())
    )
    assert all(parameter.grad is None for parameter in frozen_vicuna.parameters())


def test_binary_answer_parser() -> None:
    assert parse_binary_answer("Yes.") == "yes"
    assert parse_binary_answer("The answer is no") == "no"
    assert parse_binary_answer("unknown") is None
