"""Fast policy and cache contracts for autonomous Pixel-only training."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from spacebyte.pixel_baseline import (
    cache_metadata_matches,
    pixel_best_epoch,
    pixel_stopping_reason,
)
from spacebyte.pixel_training import (
    PermutedPixelCachedDataset,
    SpatialSensePixelCachedDataset,
    collate_pixel_features,
    feature_similarity_metrics,
    feature_similarity_passes,
    pixel_visual_cache_identity,
    require_frozen_eval,
    validate_discrete_cache_contract,
)


def metric_row(epoch, normal, zero=0.50, shuffled=0.50, macro=0.51):
    return {
        "epoch": epoch,
        "normal": {"auroc": normal, "macro_f1": macro},
        "zero": {"auroc": zero},
        "shuffled_auroc_mean": shuffled,
    }


def test_best_epoch_uses_auroc_macro_f1_and_earlier_tie() -> None:
    rows = [
        metric_row(1, 0.55, macro=0.50),
        metric_row(2, 0.55, macro=0.52),
        metric_row(3, 0.55, macro=0.52),
    ]
    assert pixel_best_epoch(rows)["epoch"] == 2


def test_autonomous_stopping_rules() -> None:
    stable = [metric_row(1, 0.53), metric_row(2, 0.525)]
    assert pixel_stopping_reason(stable) == "visual_margin_stable_two_consecutive_epochs"
    plateau = [metric_row(1, 0.510), metric_row(2, 0.511), metric_row(3, 0.5115)]
    assert pixel_stopping_reason(plateau) == "normal_auroc_plateau_two_consecutive_epochs"
    assert pixel_stopping_reason([metric_row(5, 0.51)]) == "max_epochs"
    assert pixel_stopping_reason([metric_row(1, 0.51)]) is None


def test_cache_metadata_requires_exact_identity_fields() -> None:
    expected = {"cache_version": 1, "eva_sha": "abc", "samples": 3}
    observed = {**expected, "feature_shape": [257, 1408], "created_at": "now"}
    assert cache_metadata_matches(observed, expected) == (True, [])
    matches, mismatches = cache_metadata_matches(
        {**observed, "eva_sha": "stale"}, expected
    )
    assert not matches
    assert mismatches == ["eva_sha"]


def test_pixel_cache_identity_changes_for_visual_or_weight_change() -> None:
    record = {
        "subject": {"bbox": [1, 2, 3, 4]},
        "object": {"bbox": [5, 6, 7, 8]},
        "annotation_width": 100,
        "annotation_height": 80,
    }
    first, payload = pixel_visual_cache_identity(
        record, raw_image_sha256="raw", eva_checkpoint_sha256="eva"
    )
    repeated, _ = pixel_visual_cache_identity(
        record, raw_image_sha256="raw", eva_checkpoint_sha256="eva"
    )
    changed_weight, _ = pixel_visual_cache_identity(
        record, raw_image_sha256="raw", eva_checkpoint_sha256="eva2"
    )
    changed_precision, _ = pixel_visual_cache_identity(
        record,
        raw_image_sha256="raw",
        eva_checkpoint_sha256="eva",
        cache_forward_precision="fp32",
    )
    changed_record = {**record, "subject": {"bbox": [1, 2, 3, 5]}}
    changed_box, _ = pixel_visual_cache_identity(
        changed_record, raw_image_sha256="raw", eva_checkpoint_sha256="eva"
    )
    assert first == repeated
    assert first not in {changed_weight, changed_box, changed_precision}
    assert payload["feature_location"].startswith("EVA-CLIP-G")


def write_cache(tmp_path: Path):
    rows = []
    for index in range(3):
        key = f"key-{index}"
        relative = f"features/{key}.pt"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "pixel_cache_key": key,
                "vision_features": torch.full((5, 8), float(index), dtype=torch.float16),
            },
            path,
        )
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "split": "valid",
                "pixel_cache_key": key,
                "feature_relative_path": relative,
                "question": "Where?",
                "answer": "yes" if index % 2 else "no",
                "relation": "above",
                "relation_raw": "above",
                "label": bool(index % 2),
            }
        )
    manifest = tmp_path / "valid.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return manifest


def test_cached_dataset_mapping_and_full_derangement(tmp_path: Path) -> None:
    manifest = write_cache(tmp_path)
    dataset = SpatialSensePixelCachedDataset(
        manifest, tmp_path, expected_split="valid"
    )
    shuffled = PermutedPixelCachedDataset(dataset, [1, 2, 0])
    batch = collate_pixel_features([shuffled[0], shuffled[1]])
    assert batch["sample_ids"] == ["sample-0", "sample-1"]
    assert batch["source_sample_ids"] == ["sample-1", "sample-2"]
    assert batch["vision_features"].shape == (2, 5, 8)
    assert batch["vision_attention_mask"].eq(1).all()
    with pytest.raises(ValueError, match="self mapping"):
        PermutedPixelCachedDataset(dataset, [0, 2, 1])


def test_identical_features_pass_controlled_similarity() -> None:
    value = torch.randn(2, 5, 8, dtype=torch.float16)
    metrics = feature_similarity_metrics(value, value.clone())
    assert metrics["torch_equal"]
    assert metrics["max_abs_error"] == 0
    assert feature_similarity_passes(metrics)


def test_reasonable_fp16_difference_passes() -> None:
    reference = torch.randn(4, 32, 64, dtype=torch.float16)
    candidate = (reference.float() + 0.001 * torch.randn_like(reference.float())).half()
    metrics = feature_similarity_metrics(reference, candidate)
    assert not metrics["torch_equal"]
    assert feature_similarity_passes(metrics)


def test_sparse_outlier_is_reported_and_judged_by_declared_rules() -> None:
    reference = torch.ones(1, 100, 100, dtype=torch.float16)
    candidate = reference.clone()
    candidate[0, 0, 0] += 1
    metrics = feature_similarity_metrics(reference, candidate)
    assert metrics["max_abs_error"] == 1
    assert metrics["large_error_count"] == 1
    assert feature_similarity_passes(metrics)


@pytest.mark.parametrize("field", ["cache_key", "sample_id", "input_content_hash"])
def test_discrete_cache_identity_mismatch_always_fails(field: str) -> None:
    expected = {
        "sample_id": "sample-a",
        "split": "train",
        "cache_key": "key-a",
        "input_content_hash": "image-a",
        "checkpoint_sha256": "weight-a",
        "preprocessing_sha256": "preprocess-a",
        "shape": [257, 1408],
        "dtype": "torch.float16",
        "attention_mask": "all_ones_length_257",
    }
    observed = dict(expected)
    observed[field] = "wrong"
    with pytest.raises(RuntimeError, match=field):
        validate_discrete_cache_contract(observed, expected)


def test_discrete_cache_shape_mismatch_fails() -> None:
    expected = {"shape": [257, 1408], "dtype": "torch.float16"}
    with pytest.raises(RuntimeError, match="shape"):
        validate_discrete_cache_contract(
            {"shape": [256, 1408], "dtype": "torch.float16"}, expected
        )


def test_large_scale_feature_error_fails() -> None:
    reference = torch.randn(2, 32, 64, dtype=torch.float16)
    metrics = feature_similarity_metrics(reference, -reference)
    assert not feature_similarity_passes(metrics)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nonfinite_feature_fails(bad_value: float) -> None:
    reference = torch.ones(1, 8, 8, dtype=torch.float16)
    candidate = reference.clone()
    candidate[0, 0, 0] = bad_value
    metrics = feature_similarity_metrics(reference, candidate)
    assert not metrics["all_finite"]
    assert not feature_similarity_passes(metrics)


def test_cache_encoder_must_be_frozen_and_in_eval_mode() -> None:
    encoder = torch.nn.Linear(2, 2)
    encoder.requires_grad_(False)
    encoder.eval()
    require_frozen_eval(encoder)
    encoder.train()
    with pytest.raises(RuntimeError, match="eval mode"):
        require_frozen_eval(encoder)
    encoder.eval()
    encoder.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="trainable"):
        require_frozen_eval(encoder)
