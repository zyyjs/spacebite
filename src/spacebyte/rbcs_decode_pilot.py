"""Deterministic SpatialSense-valid sampling and JPEG decode audits for RBCS."""

from __future__ import annotations

import hashlib
import io
import json
import math
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .corruptions.rbcs import RBCSParams
from .data import RAW_RELATIONS, PREPROCESS_VERSION


PILOT_VERSION = "spatialsense_rbcs_decode_pilot_v1"
DEFAULT_GLOBAL_SEED = 20261016
SAMPLES_PER_RELATION = 10
REPEATS_PER_SCENARIO = 3
JPEG_START_OF_SCAN = 0xDA


def stable_corruption_seed(
    global_seed: int,
    sample_id: str,
    scenario_name: str,
    repeat_id: int,
) -> int:
    """Derive a stable local RNG seed from the required identity fields."""

    if not isinstance(global_seed, int) or isinstance(global_seed, bool):
        raise TypeError("global_seed must be an integer")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a non-empty string")
    if not isinstance(scenario_name, str) or not scenario_name:
        raise ValueError("scenario_name must be a non-empty string")
    if not isinstance(repeat_id, int) or isinstance(repeat_id, bool) or repeat_id < 0:
        raise ValueError("repeat_id must be a non-negative integer")
    encoded = json.dumps(
        [global_seed, sample_id, scenario_name, repeat_id],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_pilot_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    global_seed: int = DEFAULT_GLOBAL_SEED,
    samples_per_relation: int = SAMPLES_PER_RELATION,
) -> dict[str, Any]:
    """Select a deterministic, relation- and label-balanced valid-only pilot."""

    if samples_per_relation <= 0:
        raise ValueError("samples_per_relation must be positive")
    forbidden = sorted({str(row.get("split")) for row in records if row.get("split") != "valid"})
    if forbidden:
        raise RuntimeError(f"RBCS decode pilot accepts valid only, got {forbidden}")
    identifiers = [str(row["sample_id"]) for row in records]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Source valid records contain duplicate sample IDs")

    grouped: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        relation = str(row["relation_raw"])
        if relation not in RAW_RELATIONS:
            raise ValueError(f"Unexpected SpatialSense relation: {relation}")
        grouped[(relation, bool(row["label"]))].append(dict(row))

    rng = random.Random(global_seed)
    selected: list[dict[str, Any]] = []
    target_no = samples_per_relation // 2
    target_yes = samples_per_relation - target_no
    relation_counts: dict[str, dict[str, int]] = {}
    for relation in RAW_RELATIONS:
        candidates: dict[bool, list[dict[str, Any]]] = {}
        for label in (False, True):
            values = sorted(grouped[(relation, label)], key=lambda item: item["sample_id"])
            rng.shuffle(values)
            candidates[label] = values
        chosen_no = candidates[False][:target_no]
        chosen_yes = candidates[True][:target_yes]
        chosen_ids = {item["sample_id"] for item in chosen_no + chosen_yes}
        if len(chosen_ids) < samples_per_relation:
            remainder = [
                item
                for label in (False, True)
                for item in candidates[label]
                if item["sample_id"] not in chosen_ids
            ]
            rng.shuffle(remainder)
            needed = samples_per_relation - len(chosen_ids)
            chosen_extra = remainder[:needed]
        else:
            chosen_extra = []
        chosen = chosen_no + chosen_yes + chosen_extra
        if len(chosen) != samples_per_relation:
            raise RuntimeError(
                f"Not enough valid samples for {relation}: {len(chosen)}/{samples_per_relation}"
            )
        chosen.sort(key=lambda item: (bool(item["label"]), item["sample_id"]))
        selected.extend(chosen)
        relation_counts[relation] = {
            "total": len(chosen),
            "yes": sum(bool(item["label"]) for item in chosen),
            "no": sum(not bool(item["label"]) for item in chosen),
        }

    selected.sort(
        key=lambda item: (
            RAW_RELATIONS.index(item["relation_raw"]),
            bool(item["label"]),
            item["sample_id"],
        )
    )
    selected_ids = [item["sample_id"] for item in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("Selected pilot contains duplicate sample IDs")
    identity = {
        "pilot_version": PILOT_VERSION,
        "global_seed": global_seed,
        "source_split": "valid",
        "samples_per_relation": samples_per_relation,
        "relation_order": list(RAW_RELATIONS),
        "sample_ids": selected_ids,
        "preprocessing_version": PREPROCESS_VERSION,
    }
    return {
        **identity,
        "selection_sha256": _canonical_sha256(identity),
        "sample_count": len(selected),
        "relation_counts": relation_counts,
        "samples": selected,
        "test_split_read": False,
    }


def _warning_rows(caught: Sequence[warnings.WarningMessage], phase: str) -> list[dict[str, str]]:
    return [
        {
            "phase": phase,
            "category": item.category.__name__,
            "message": str(item.message),
        }
        for item in caught
    ]


def decode_jpeg_bytes(payload: bytes) -> dict[str, Any]:
    """Run independent PIL verify, full load, and RGB/finite checks."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    result: dict[str, Any] = {
        "verify_success": False,
        "load_success": False,
        "rgb_success": False,
        "decoded_width": None,
        "decoded_height": None,
        "exception_type": None,
        "exception_message": None,
        "failure_stage": None,
        "verify_exception_type": None,
        "verify_exception_message": None,
        "load_exception_type": None,
        "load_exception_message": None,
        "rgb_exception_type": None,
        "rgb_exception_message": None,
        "warnings": [],
        "replacement_used": False,
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with Image.open(io.BytesIO(payload)) as image:
                image.verify()
            result["verify_success"] = True
        except Exception as error:
            result["verify_exception_type"] = type(error).__name__
            result["verify_exception_message"] = str(error)
    result["warnings"].extend(_warning_rows(caught, "verify"))

    image = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            image = Image.open(io.BytesIO(payload))
            image.load()
            result["load_success"] = True
            result["decoded_width"], result["decoded_height"] = image.size
        except Exception as error:
            result["load_exception_type"] = type(error).__name__
            result["load_exception_message"] = str(error)
    result["warnings"].extend(_warning_rows(caught, "load"))

    if result["load_success"] and image is not None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                rgb = image.convert("RGB")
                pixels = np.asarray(rgb)
                if pixels.shape != (
                    result["decoded_height"],
                    result["decoded_width"],
                    3,
                ):
                    raise RuntimeError(f"Unexpected RGB array shape: {pixels.shape}")
                if not bool(np.isfinite(pixels).all()):
                    raise FloatingPointError("Decoded RGB pixels contain NaN or Inf")
                result["rgb_success"] = True
            except Exception as error:
                result["rgb_exception_type"] = type(error).__name__
                result["rgb_exception_message"] = str(error)
        result["warnings"].extend(_warning_rows(caught, "rgb"))
    if image is not None:
        image.close()

    for stage in ("verify", "load", "rgb"):
        error_type = result[f"{stage}_exception_type"]
        if error_type is not None:
            result["exception_type"] = error_type
            result["exception_message"] = result[f"{stage}_exception_message"]
            result["failure_stage"] = stage
            break
    return result


def _jpeg_header_length(payload: bytes) -> int:
    """Return the end of the JPEG SOS header, before entropy-coded bytes."""

    if len(payload) < 4 or not payload.startswith(b"\xff\xd8"):
        raise ValueError("Original stream is not a JPEG with an SOI marker")
    offset = 2
    while offset < len(payload):
        if payload[offset] != 0xFF:
            raise ValueError(f"Malformed JPEG marker at byte {offset}")
        while offset < len(payload) and payload[offset] == 0xFF:
            offset += 1
        if offset >= len(payload):
            raise ValueError("JPEG ended inside marker prefix")
        marker = payload[offset]
        offset += 1
        if marker in {0x01, *range(0xD0, 0xD9)}:
            continue
        if marker == 0xD9:
            return offset
        if offset + 2 > len(payload):
            raise ValueError("JPEG ended before marker length")
        segment_length = int.from_bytes(payload[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(payload):
            raise ValueError(f"Invalid JPEG segment length at byte {offset}")
        offset += segment_length
        if marker == JPEG_START_OF_SCAN:
            return offset
    raise ValueError("JPEG does not contain a Start of Scan marker")


def structural_status(original: bytes, corrupted: bytes) -> dict[str, Any]:
    header_length = _jpeg_header_length(original)
    return {
        "soi_preserved": bool(
            original.startswith(b"\xff\xd8") and corrupted.startswith(b"\xff\xd8")
        ),
        "eoi_preserved": bool(
            original.endswith(b"\xff\xd9") and corrupted.endswith(b"\xff\xd9")
        ),
        "jpeg_header_length": header_length,
        "header_bytes_modified": original[:header_length] != corrupted[:header_length],
    }


def scenario_classification(rgb_success_rate: float) -> str:
    if not math.isfinite(rgb_success_rate) or not 0.0 <= rgb_success_rate <= 1.0:
        raise ValueError("rgb_success_rate must be finite and in [0, 1]")
    if rgb_success_rate >= 0.8:
        return "paired_evaluation_ready"
    if rgb_success_rate >= 0.3:
        return "partially_decodable"
    if rgb_success_rate > 0.0:
        return "fallback_only"
    return "pixel_unavailable"


def summarize_results(
    rows: Sequence[Mapping[str, Any]],
    params_by_scenario: Mapping[str, RBCSParams | None],
) -> list[dict[str, Any]]:
    """Aggregate every success and failure by scenario and repeat."""

    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scenario_name"]), int(row["repeat_id"]))].append(row)
    summaries = []
    for (scenario, repeat), values in sorted(grouped.items()):
        count = len(values)
        if count == 0:
            raise RuntimeError("Summary group unexpectedly has zero rows")
        params = params_by_scenario[scenario]
        theoretical_total = float(params.P * params.q) if params else 0.0
        theoretical_flip = float(params.P * params.q * params.p_f) if params else 0.0
        theoretical_delete = (
            float(params.P * params.q * (1.0 - params.p_f)) if params else 0.0
        )
        exception_counts = Counter(
            str(row["decode"]["exception_type"] or "none") for row in values
        )
        mean = lambda key: sum(float(row["corruption"][key]) for row in values) / count
        verify_rate = sum(bool(row["decode"]["verify_success"]) for row in values) / count
        load_rate = sum(bool(row["decode"]["load_success"]) for row in values) / count
        rgb_rate = sum(bool(row["decode"]["rgb_success"]) for row in values) / count
        summaries.append(
            {
                "scenario_name": scenario,
                "repeat_id": repeat,
                "sample_count": count,
                "mean_original_length": mean("original_length"),
                "mean_corrupted_length": mean("corrupted_length"),
                "mean_flipped_byte_count": mean("flipped_byte_count"),
                "mean_flip_rate": mean("flip_rate"),
                "mean_deleted_byte_count": mean("deleted_byte_count"),
                "mean_deletion_rate": mean("deletion_rate"),
                "mean_modified_byte_count": mean("modified_byte_count"),
                "mean_total_modified_rate": mean("total_modified_rate"),
                "theoretical_total_modified_rate": theoretical_total,
                "total_modified_rate_deviation": mean("total_modified_rate")
                - theoretical_total,
                "theoretical_flip_rate": theoretical_flip,
                "flip_rate_deviation": mean("flip_rate") - theoretical_flip,
                "theoretical_deletion_rate": theoretical_delete,
                "deletion_rate_deviation": mean("deletion_rate")
                - theoretical_delete,
                "verify_success_rate": verify_rate,
                "load_success_rate": load_rate,
                "rgb_success_rate": rgb_rate,
                "exception_type_distribution": dict(sorted(exception_counts.items())),
                "soi_preserved_rate": sum(
                    bool(row["structure"]["soi_preserved"]) for row in values
                )
                / count,
                "eoi_preserved_rate": sum(
                    bool(row["structure"]["eoi_preserved"]) for row in values
                )
                / count,
                "header_modified_rate": sum(
                    bool(row["structure"]["header_bytes_modified"]) for row in values
                )
                / count,
                "classification": scenario_classification(rgb_rate),
            }
        )
    return summaries


def aggregate_scenarios(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scenario_name"])].append(row)
    result = []
    for scenario, values in grouped.items():
        count = len(values)
        rgb_rate = sum(bool(item["decode"]["rgb_success"]) for item in values) / count
        result.append(
            {
                "scenario_name": scenario,
                "sample_count": count,
                "verify_success_rate": sum(
                    bool(item["decode"]["verify_success"]) for item in values
                )
                / count,
                "load_success_rate": sum(
                    bool(item["decode"]["load_success"]) for item in values
                )
                / count,
                "rgb_success_rate": rgb_rate,
                "mean_flip_rate": sum(
                    float(item["corruption"]["flip_rate"]) for item in values
                )
                / count,
                "mean_deletion_rate": sum(
                    float(item["corruption"]["deletion_rate"]) for item in values
                )
                / count,
                "mean_total_modified_rate": sum(
                    float(item["corruption"]["total_modified_rate"]) for item in values
                )
                / count,
                "classification": scenario_classification(rgb_rate),
            }
        )
    order = {"Clean": -1, **{name: index for index, name in enumerate(params_name_order())}}
    result.sort(key=lambda item: order[item["scenario_name"]])
    return result


def params_name_order() -> tuple[str, ...]:
    return tuple(
        f"{level}-{kind}"
        for level in ("Light", "Medium", "Heavy", "Extreme")
        for kind in ("Flip", "Loss", "Mixed")
    )
