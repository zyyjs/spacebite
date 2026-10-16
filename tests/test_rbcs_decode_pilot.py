"""Policy, determinism, decoding, and aggregation tests for the RBCS pilot."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from spacebyte.corruptions.rbcs import RBCSCorruptor, RBCSParams, SCENARIOS
from spacebyte.rbcs_decode_pilot import (
    aggregate_scenarios,
    build_pilot_manifest,
    decode_jpeg_bytes,
    scenario_classification,
    stable_corruption_seed,
    structural_status,
    summarize_results,
)


def fake_records() -> list[dict]:
    relations = (
        "above",
        "behind",
        "in",
        "in front of",
        "next to",
        "on",
        "to the left of",
        "to the right of",
        "under",
    )
    rows = []
    for relation_index, relation in enumerate(relations):
        for label in (False, True):
            for index in range(8):
                rows.append(
                    {
                        "sample_id": f"{relation_index}-{int(label)}-{index}",
                        "split": "valid",
                        "relation_raw": relation,
                        "label": label,
                    }
                )
    return rows


def test_manifest_is_fixed_balanced_unique_and_valid_only() -> None:
    first = build_pilot_manifest(fake_records(), global_seed=123)
    second = build_pilot_manifest(fake_records(), global_seed=123)
    assert first == second
    assert first["sample_count"] == 90
    assert len({row["sample_id"] for row in first["samples"]}) == 90
    assert all(value == {"total": 10, "yes": 5, "no": 5} for value in first["relation_counts"].values())
    assert first["test_split_read"] is False


def test_manifest_rejects_any_test_record() -> None:
    rows = fake_records()
    rows[0] = {**rows[0], "split": "test"}
    with pytest.raises(RuntimeError, match="valid only"):
        build_pilot_manifest(rows)


def test_required_seed_fields_are_stable_and_affect_seed() -> None:
    base = stable_corruption_seed(7, "sample", "Heavy-Mixed", 0)
    assert base == stable_corruption_seed(7, "sample", "Heavy-Mixed", 0)
    alternatives = {
        stable_corruption_seed(8, "sample", "Heavy-Mixed", 0),
        stable_corruption_seed(7, "sample-2", "Heavy-Mixed", 0),
        stable_corruption_seed(7, "sample", "Heavy-Flip", 0),
        stable_corruption_seed(7, "sample", "Heavy-Mixed", 1),
    }
    assert base not in alternatives and len(alternatives) == 4
    data = bytes(range(256)) * 4
    assert RBCSCorruptor.corrupt(data, SCENARIOS["Heavy-Mixed"], base) == RBCSCorruptor.corrupt(
        data, SCENARIOS["Heavy-Mixed"], base
    )


def jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (20, 12), (10, 90, 170)).save(buffer, format="JPEG", quality=100)
    return buffer.getvalue()


def test_verify_load_and_rgb_are_separate_fields() -> None:
    result = decode_jpeg_bytes(jpeg_bytes())
    assert result["verify_success"] is True
    assert result["load_success"] is True
    assert result["rgb_success"] is True
    assert result["decoded_width"] == 20 and result["decoded_height"] == 12
    assert result["replacement_used"] is False


def test_decode_failure_is_explicit_and_never_replaced() -> None:
    result = decode_jpeg_bytes(b"not a jpeg")
    assert result["verify_success"] is False
    assert result["load_success"] is False
    assert result["rgb_success"] is False
    assert result["exception_type"] == "UnidentifiedImageError"
    assert result["replacement_used"] is False
    assert result["decoded_width"] is None and result["decoded_height"] is None


def synthetic_row(success: bool, scenario="Heavy-Mixed", repeat=0) -> dict:
    return {
        "scenario_name": scenario,
        "repeat_id": repeat,
        "corruption": {
            "original_length": 100,
            "corrupted_length": 90,
            "flipped_byte_count": 10,
            "flip_rate": 0.1,
            "deleted_byte_count": 10,
            "deletion_rate": 0.1,
            "modified_byte_count": 20,
            "total_modified_rate": 0.2,
        },
        "decode": {
            "verify_success": success,
            "load_success": success,
            "rgb_success": success,
            "exception_type": None if success else "OSError",
        },
        "structure": {
            "soi_preserved": success,
            "eoi_preserved": success,
            "header_bytes_modified": not success,
        },
    }


def test_summary_denominator_includes_all_failures() -> None:
    rows = [synthetic_row(True), synthetic_row(False), synthetic_row(False)]
    summary = summarize_results(rows, {"Heavy-Mixed": SCENARIOS["Heavy-Mixed"]})[0]
    assert summary["sample_count"] == 3
    assert summary["verify_success_rate"] == pytest.approx(1 / 3)
    assert summary["load_success_rate"] == pytest.approx(1 / 3)
    assert summary["rgb_success_rate"] == pytest.approx(1 / 3)
    aggregate = aggregate_scenarios(rows)[0]
    assert aggregate["sample_count"] == 3
    assert aggregate["rgb_success_rate"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("rate", "expected"),
    [
        (1.0, "paired_evaluation_ready"),
        (0.8, "paired_evaluation_ready"),
        (0.7999, "partially_decodable"),
        (0.3, "partially_decodable"),
        (0.2999, "fallback_only"),
        (0.0001, "fallback_only"),
        (0.0, "pixel_unavailable"),
    ],
)
def test_scenario_classification_boundaries(rate, expected) -> None:
    assert scenario_classification(rate) == expected


def test_structure_checks_soi_eoi_and_complete_jpeg_header() -> None:
    original = jpeg_bytes()
    clean = structural_status(original, original)
    assert clean["soi_preserved"] is True
    assert clean["eoi_preserved"] is True
    assert clean["header_bytes_modified"] is False
    assert 64 < clean["jpeg_header_length"] < len(original)
    corrupted = bytes([original[0] ^ 1]) + original[1:-2] + b"xx"
    status = structural_status(original, corrupted)
    assert not status["soi_preserved"]
    assert not status["eoi_preserved"]
    assert status["header_bytes_modified"]


def test_summary_reports_theory_and_deviation() -> None:
    row = synthetic_row(True, scenario="Heavy-Mixed")
    summary = summarize_results([row], {"Heavy-Mixed": RBCSParams(32, 0.65, 0.5, 0.65)})[0]
    assert summary["theoretical_total_modified_rate"] == pytest.approx(0.4225)
    assert summary["theoretical_flip_rate"] == pytest.approx(0.21125)
    assert summary["theoretical_deletion_rate"] == pytest.approx(0.21125)
    assert summary["total_modified_rate_deviation"] == pytest.approx(0.2 - 0.4225)
