"""Fast contracts for the full SpatialSense cache, evaluation, and pilot."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from spacebyte.data import (
    PREPROCESS_VERSION,
    SpatialSenseFullCachedDataset,
    VISUAL_PREPROCESS_CONFIG,
    atomic_torch_save,
    collate_cached_features,
    validate_feature_cache_file,
    visual_cache_identity,
)
from spacebyte.evaluation import (
    binary_classification_metrics,
    conditional_log_likelihood,
    fixed_derangement,
    greedy_metrics,
    likelihood_metrics,
    parse_greedy_binary,
)
from spacebyte.full_pipeline import PermutedVisualDataset
from spacebyte.models import (
    build_xinstruct_vicuna_tokenizer,
    scale_projected_queries,
)
from spacebyte.training import checkpoint_is_better, warmup_cosine_multiplier


VICUNA_PATH = Path(
    "/data/students/Zhang_jinsong/spacebyte/models/vicuna-7b-v1.1"
)


def record(sample_id: str, split: str, key: str, answer: str = "yes") -> dict:
    return {
        "sample_id": sample_id,
        "split": split,
        "visual_cache_key": key,
        "cache_relative_path": f"features/{key}.pt",
        "question": "Is red above blue? Answer yes or no.",
        "answer": answer,
        "relation": "above",
        "relation_raw": "above",
        "label": answer == "yes",
    }


def write_cache(root: Path, key: str, length: int) -> None:
    atomic_torch_save(
        {
            "sample_id": key,
            "split": "train",
            "visual_cache_key": key,
            "byte_features": torch.arange(length * 192).reshape(length, 192).half(),
            "byte_attention_mask": torch.ones(length, dtype=torch.uint8),
            "valid_token_count": length,
            "jpeg_byte_length": 100,
            "processed_jpeg_sha256": "b" * 64,
            "preprocessing_version": PREPROCESS_VERSION,
        },
        root / f"features/{key}.pt",
    )


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_split_isolation_rejects_mixed_manifest(tmp_path) -> None:
    manifest = tmp_path / "mixed.jsonl"
    write_manifest(manifest, [record("a", "train", "a"), record("b", "valid", "b")])
    with pytest.raises(RuntimeError, match="outside"):
        SpatialSenseFullCachedDataset(manifest, tmp_path, expected_split="train")


def test_visual_cache_key_contains_boxes_preprocessing_and_checkpoint() -> None:
    base = {
        "subject": {"bbox": [1, 2, 3, 4]},
        "object": {"bbox": [5, 6, 7, 8]},
    }
    key, payload = visual_cache_identity(
        base, raw_image_sha256="a" * 64, byteformer_checkpoint_sha256="b" * 64
    )
    moved = {**base, "subject": {"bbox": [1, 2, 3, 5]}}
    moved_key, _ = visual_cache_identity(
        moved, raw_image_sha256="a" * 64, byteformer_checkpoint_sha256="b" * 64
    )
    other_checkpoint, _ = visual_cache_identity(
        base, raw_image_sha256="a" * 64, byteformer_checkpoint_sha256="c" * 64
    )
    assert key != moved_key != other_checkpoint
    assert payload["subject_bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert payload["object_bbox"] == [5.0, 6.0, 7.0, 8.0]
    assert payload["preprocessing"] == VISUAL_PREPROCESS_CONFIG
    assert payload["preprocessing"]["jpeg_quality"] == 100


def test_full_cache_mapping_count_and_padding(tmp_path) -> None:
    write_cache(tmp_path, "key-a", 2)
    write_cache(tmp_path, "key-b", 4)
    rows = [record("a", "train", "key-a"), record("b", "train", "key-b", "no")]
    manifest = tmp_path / "train.jsonl"
    write_manifest(manifest, rows)
    dataset = SpatialSenseFullCachedDataset(
        manifest, tmp_path, expected_split="train"
    )
    assert len(dataset) == 2
    assert {dataset[index]["sample_id"] for index in range(2)} == {"a", "b"}
    batch = collate_cached_features([dataset[0], dataset[1]])
    assert batch["byte_features"].shape == (2, 4, 192)
    assert batch["byte_attention_mask"].tolist() == [[1, 1, 0, 0], [1, 1, 1, 1]]
    assert batch["splits"] == ["train", "train"]


def test_cache_resume_validation_and_corruption_detection(tmp_path) -> None:
    path = tmp_path / "features/good.pt"
    write_cache(tmp_path, "good", 3)
    assert validate_feature_cache_file(path, expected_visual_cache_key="good") == (
        True,
        "valid",
    )
    assert validate_feature_cache_file(path, expected_visual_cache_key="wrong")[1] == "cache_key_mismatch"
    path.write_bytes(b"truncated")
    valid, reason = validate_feature_cache_file(path, expected_visual_cache_key="good")
    assert not valid and reason.startswith("unreadable:")


def test_yes_no_tokenization_has_equal_answer_eos_length() -> None:
    if not VICUNA_PATH.joinpath("tokenizer.model").exists():
        pytest.skip("Pinned Vicuna tokenizer is unavailable")
    tokenizer = build_xinstruct_vicuna_tokenizer(VICUNA_PATH)
    rows = {}
    for answer in ("yes", "no"):
        encoded = tokenizer(
            f"{answer}{tokenizer.eos_token}", add_special_tokens=True
        ).input_ids[1:]
        rows[answer] = encoded
        assert encoded[-1] == tokenizer.eos_token_id
    assert len(rows["yes"]) == len(rows["no"]) == 2


def test_conditional_likelihood_uses_causal_shift() -> None:
    labels = torch.tensor([[-100, -100, 2, 3], [-100, -100, 4, 5]])
    logits = torch.zeros(2, 4, 8)
    logits[0, 1, 2] = 8
    logits[0, 2, 3] = 8
    logits[1, 1, 4] = 2
    logits[1, 2, 5] = 2
    sums, counts = conditional_log_likelihood(logits, labels)
    assert counts.tolist() == [2, 2]
    assert sums[0] > sums[1]


def test_greedy_parser_is_strict_and_metrics_count_invalid() -> None:
    assert parse_greedy_binary("Yes.") == "yes"
    assert parse_greedy_binary(" no ") == "no"
    assert parse_greedy_binary("The answer is yes") == "invalid"
    result = greedy_metrics(
        [True, False, True], ["yes", "no", "invalid"], ["above"] * 3
    )
    assert result["exact_match_accuracy"] == pytest.approx(2 / 3)
    assert result["invalid_rate"] == pytest.approx(1 / 3)
    assert result["confusion_matrix"]["yes"]["invalid"] == 1


def test_likelihood_metrics_and_tied_auroc() -> None:
    result = likelihood_metrics(
        [False, True, False, True], [-2.0, 2.0, -1.0, 1.0], ["above"] * 4
    )
    assert result["accuracy"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["f1"] == 1.0
    assert result["auroc"] == 1.0


def test_zero_byte_scale_sets_only_projected_queries_to_zero() -> None:
    projected = torch.randn(2, 32, 4096)
    zero = scale_projected_queries(projected, 0.0)
    assert torch.count_nonzero(zero) == 0
    assert torch.equal(scale_projected_queries(projected, 1.0), projected)


def test_shuffled_byte_has_no_self_mapping_and_replaces_only_visual(tmp_path) -> None:
    for key, length in (("a", 2), ("b", 3), ("c", 4)):
        write_cache(tmp_path, key, length)
    rows = [record(key, "valid", key, "yes" if key != "b" else "no") for key in "abc"]
    manifest = tmp_path / "valid.jsonl"
    write_manifest(manifest, rows)
    dataset = SpatialSenseFullCachedDataset(
        manifest, tmp_path, expected_split="valid"
    )
    mapping = fixed_derangement(3, 20261013, ["a", "b", "c"])
    assert all(index != source for index, source in enumerate(mapping))
    shuffled = PermutedVisualDataset(dataset, mapping)
    assert shuffled[0]["sample_id"] == dataset[0]["sample_id"]
    assert shuffled[0]["answer"] == dataset[0]["answer"]
    assert shuffled[0]["byte_features"].shape == dataset[mapping[0]]["byte_features"].shape
    duplicate_visual_mapping = fixed_derangement(
        6, 20261013, ["shared", "shared", "b", "c", "d", "e"]
    )
    keys = ["shared", "shared", "b", "c", "d", "e"]
    assert all(keys[index] != keys[source] for index, source in enumerate(duplicate_visual_mapping))


def test_valid_and_test_manifest_cannot_be_opened_as_train(tmp_path) -> None:
    for split in ("valid", "test"):
        manifest = tmp_path / f"{split}.jsonl"
        write_manifest(manifest, [record(split, split, split)])
        with pytest.raises(RuntimeError):
            SpatialSenseFullCachedDataset(manifest, tmp_path, expected_split="train")


def test_checkpoint_state_strict_reload_and_selection(tmp_path) -> None:
    source = torch.nn.Linear(3, 2)
    path = tmp_path / "bridge.pt"
    atomic_torch_save({"bridge_state_dict": source.state_dict()}, path)
    target = torch.nn.Linear(3, 2)
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    incompatible = target.load_state_dict(artifact["bridge_state_dict"], strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert all(torch.equal(a, b) for a, b in zip(source.parameters(), target.parameters()))
    assert checkpoint_is_better(0.6, 0.7, 0.5, 0.9)
    assert checkpoint_is_better(0.6, 0.8, 0.6, 0.7)
    assert not checkpoint_is_better(0.6, 0.6, 0.6, 0.7)


def test_warmup_cosine_schedule_boundaries() -> None:
    assert warmup_cosine_multiplier(0, warmup_steps=100, total_steps=1405) == 0.01
    assert warmup_cosine_multiplier(99, warmup_steps=100, total_steps=1405) == 1.0
    assert warmup_cosine_multiplier(100, warmup_steps=100, total_steps=1405) == 1.0
    assert warmup_cosine_multiplier(1405, warmup_steps=100, total_steps=1405) == 0.0
