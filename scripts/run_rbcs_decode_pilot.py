#!/usr/bin/env python3
"""Run the CPU-only SpatialSense valid RBCS JPEG-decodability pilot."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from spacebyte.corruptions.rbcs import (
    ALGORITHM_NAME,
    ALGORITHM_VERSION,
    RBCSCorruptor,
    SCENARIOS,
)
from spacebyte.data import (
    flatten_spatialsense_annotations,
    render_record_to_jpeg_bytes,
    sha256_file,
)
from spacebyte.rbcs_decode_pilot import (
    DEFAULT_GLOBAL_SEED,
    PILOT_VERSION,
    REPEATS_PER_SCENARIO,
    aggregate_scenarios,
    build_pilot_manifest,
    decode_jpeg_bytes,
    stable_corruption_seed,
    structural_status,
    summarize_results,
)


DATA_ROOT = Path("/data/students/Zhang_jinsong/spacebyte")
DEFAULT_OUTPUT = DATA_ROOT / "result/rbcs/decode_pilot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DATA_ROOT / "dataset/SpatialSense")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--global-seed", type=int, default=DEFAULT_GLOBAL_SEED)
    parser.add_argument("--repeats", type=int, default=REPEATS_PER_SCENARIO)
    return parser.parse_args()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: Any) -> None:
    atomic_bytes(
        path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )


def clean_statistics(payload: bytes) -> dict[str, Any]:
    import hashlib

    digest = hashlib.sha256(payload).hexdigest()
    return {
        "algorithm_name": "Clean",
        "algorithm_version": "clean",
        "algorithm_source": "project shared marked-JPEG preprocessing",
        "seed": None,
        "scenario_name": "Clean",
        "parameters": None,
        "original_length": len(payload),
        "corrupted_length": len(payload),
        "complete_segment_count": 0,
        "selected_segment_count": 0,
        "flip_segment_count": 0,
        "loss_segment_count": 0,
        "flipped_byte_count": 0,
        "deleted_byte_count": 0,
        "modified_byte_count": 0,
        "flip_rate": 0.0,
        "deletion_rate": 0.0,
        "total_modified_rate": 0.0,
        "original_sha256": digest,
        "corrupted_sha256": digest,
    }


def result_row(
    record: Mapping[str, Any],
    scenario: str,
    repeat_id: int,
    seed: int | None,
    original: bytes,
    payload: bytes,
    corruption: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "sample_id": record["sample_id"],
        "split": record["split"],
        "relation": record["relation_raw"],
        "label": bool(record["label"]),
        "scenario_name": scenario,
        "repeat_id": repeat_id,
        "corruption_seed": seed,
        "corruption": dict(corruption),
        "structure": structural_status(original, payload),
        "decode": decode_jpeg_bytes(payload),
    }


def maybe_save_example(
    output_dir: Path,
    registry: dict[str, set[str]],
    level: str,
    payload: bytes,
    row: Mapping[str, Any],
) -> None:
    kind = "success" if row["decode"]["rgb_success"] else "failure"
    if kind in registry[level]:
        return
    registry[level].add(kind)
    safe_scenario = str(row["scenario_name"]).lower().replace("-", "_")
    stem = (
        f"{level.lower()}_{kind}_{row['sample_id']}_{safe_scenario}"
        f"_r{row['repeat_id']}"
    )
    example_dir = output_dir / "examples"
    jpg_path = example_dir / f"{stem}.jpg"
    json_path = example_dir / f"{stem}.json"
    atomic_bytes(jpg_path, payload)
    example = {**dict(row), "corrupted_jpeg_path": str(jpg_path.resolve())}
    if kind == "success":
        png_path = example_dir / f"{stem}.png"
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            image.convert("RGB").save(png_path, format="PNG")
        example["png_preview_path"] = str(png_path.resolve())
    else:
        example["png_preview_path"] = None
    write_json(json_path, example)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "scenario_name",
        "repeat_id",
        "sample_count",
        "mean_original_length",
        "mean_corrupted_length",
        "mean_flipped_byte_count",
        "mean_flip_rate",
        "mean_deleted_byte_count",
        "mean_deletion_rate",
        "mean_modified_byte_count",
        "mean_total_modified_rate",
        "theoretical_total_modified_rate",
        "total_modified_rate_deviation",
        "theoretical_flip_rate",
        "flip_rate_deviation",
        "theoretical_deletion_rate",
        "deletion_rate_deviation",
        "verify_success_rate",
        "load_success_rate",
        "rgb_success_rate",
        "soi_preserved_rate",
        "eoi_preserved_rate",
        "header_modified_rate",
        "classification",
        "exception_type_distribution",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: (
                            json.dumps(row[key], ensure_ascii=False, sort_keys=True)
                            if key == "exception_type_distribution"
                            else row[key]
                        )
                        for key in fields
                    }
                )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def render_markdown(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_file_sha256: str,
    scenario_rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    exceptions: Mapping[str, Any],
    examples: Mapping[str, set[str]],
    conclusion: str,
    output_dir: Path,
) -> None:
    aggregate_lines = [
        "| Scenario | Actual flip | Theory flip | Actual delete | Theory delete | Actual total | Theory total | Verify | Load | RGB | Classification |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    params = {"Clean": None, **SCENARIOS}
    for row in scenario_rows:
        scenario = row["scenario_name"]
        parameter = params[scenario]
        theory = float(parameter.P * parameter.q) if parameter else 0.0
        theory_flip = theory * float(parameter.p_f) if parameter else 0.0
        theory_delete = theory * (1.0 - float(parameter.p_f)) if parameter else 0.0
        aggregate_lines.append(
            f"| {scenario} | {row['mean_flip_rate']:.4f} | {theory_flip:.4f} "
            f"| {row['mean_deletion_rate']:.4f} | {theory_delete:.4f} "
            f"| {row['mean_total_modified_rate']:.4f} | {theory:.4f} "
            f"| {row['verify_success_rate']:.1%} | {row['load_success_rate']:.1%} "
            f"| {row['rgb_success_rate']:.1%} | {row['classification']} |"
        )
    repeat_lines = [
        "| Scenario | Repeat | N | Modified | Δ theory | Load | RGB | SOI | EOI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        repeat_lines.append(
            f"| {row['scenario_name']} | {row['repeat_id']} | {row['sample_count']} "
            f"| {row['mean_total_modified_rate']:.4f} "
            f"| {row['total_modified_rate_deviation']:+.4f} "
            f"| {row['load_success_rate']:.1%} | {row['rgb_success_rate']:.1%} "
            f"| {row['soi_preserved_rate']:.1%} | {row['eoi_preserved_rate']:.1%} |"
        )
    exception_lines = []
    for scenario in [row["scenario_name"] for row in scenario_rows]:
        exception_lines.append(
            f"- **{scenario}**: `{json.dumps(exceptions['by_scenario'][scenario], ensure_ascii=False, sort_keys=True)}`"
        )
    example_lines = [
        f"- {level}: saved categories `{', '.join(sorted(kinds)) or 'none'}` under `{(output_dir / 'examples').resolve()}`"
        for level, kinds in examples.items()
    ]
    text = f"""# SpatialSense RBCS Decodability Pilot

## Data selection

- Source split: SpatialSense **valid only**; test was not read.
- Selection: {manifest['sample_count']} unique samples, 10 per each of nine relations, with 5 yes / 5 no wherever available.
- Global seed: `{manifest['global_seed']}`
- Selection SHA-256: `{manifest['selection_sha256']}`
- Manifest file SHA-256: `{manifest_file_sha256}`
- Preprocessing: shared red-subject/blue-object, 224×224 letterbox, JPEG Q100 pipeline (`{manifest['preprocessing_version']}`).

## RBCS setup

- `{ALGORITHM_NAME}` / `{ALGORITHM_VERSION}` from `spacebyte.corruptions.rbcs`.
- Clean plus all 12 canonical scenarios; each corrupted sample has {REPEATS_PER_SCENARIO} deterministic repeats.
- Corruption seeds are derived from global seed, sample ID, scenario, and repeat ID.
- Full-stream corruption is used without header preservation, repair, truncated-image loading, or decode substitution.

The theoretical total modified fraction over complete segments is `P × q`:
Light 0.1225, Medium 0.3025, Heavy 0.4225, Extreme 0.5625. For Mixed,
theory splits into `P × q × p_f` flips and `P × q × (1-p_f)` deletions.
Finite samples, segment-level operation selection, and unprocessed tail bytes explain deviations.

## Scenario results (three repeats combined)

{chr(10).join(aggregate_lines)}

## Per-repeat theory comparison and structural markers

{chr(10).join(repeat_lines)}

## Main exception types

{chr(10).join(exception_lines)}

## Representative examples

{chr(10).join(example_lines)}

Failed samples retain their corrupted JPEG and JSON only; no clean, black, or zero image replaces them. Successful samples additionally have a PNG preview.

## Conclusion

Non-Clean scenarios with RGB success rate ≥30%: **{sum(row['scenario_name'] != 'Clean' and row['rgb_success_rate'] >= 0.3 for row in scenario_rows)}**.

Final decision: **`{conclusion}`**.

This CPU pilot measures only corruption and decodability. It does not load or evaluate EVA, ByteFormer, either Q-Former, or Vicuna.
"""
    atomic_bytes(path, text.encode("utf-8"))


def main() -> int:
    args = parse_args()
    if args.repeats != REPEATS_PER_SCENARIO:
        raise ValueError(f"Canonical pilot requires exactly {REPEATS_PER_SCENARIO} repeats")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = flatten_spatialsense_annotations(
        args.dataset_root, include_splits={"valid"}
    )
    manifest = build_pilot_manifest(records, global_seed=args.global_seed)
    manifest_path = args.output_dir / "pilot_manifest.json"
    write_json(manifest_path, manifest)
    manifest_file_sha256 = sha256_file(manifest_path)
    atomic_bytes(
        args.output_dir / "pilot_manifest.sha256",
        f"{manifest_file_sha256}  pilot_manifest.json\n".encode("ascii"),
    )

    results_path = args.output_dir / "per_sample_results.jsonl"
    temporary_results = results_path.with_name(f".{results_path.name}.tmp-{os.getpid()}")
    all_rows: list[dict[str, Any]] = []
    example_registry = {
        level: set() for level in ("Light", "Medium", "Heavy", "Extreme")
    }
    try:
        with temporary_results.open("w", encoding="utf-8") as handle:
            for record in manifest["samples"]:
                byte_tensor, _image, _boxes = render_record_to_jpeg_bytes(
                    args.dataset_root, record
                )
                original = byte_tensor.to(dtype=byte_tensor.dtype, device="cpu").numpy().astype(
                    "uint8", copy=False
                ).tobytes()
                clean = result_row(
                    record,
                    "Clean",
                    0,
                    None,
                    original,
                    original,
                    clean_statistics(original),
                )
                all_rows.append(clean)
                handle.write(json.dumps(clean, ensure_ascii=False) + "\n")
                for scenario, params in SCENARIOS.items():
                    level = scenario.split("-", 1)[0]
                    for repeat_id in range(args.repeats):
                        seed = stable_corruption_seed(
                            args.global_seed,
                            record["sample_id"],
                            scenario,
                            repeat_id,
                        )
                        corrupted, statistics = RBCSCorruptor.corrupt_with_stats(
                            original,
                            params,
                            seed=seed,
                            scenario_name=scenario,
                        )
                        row = result_row(
                            record,
                            scenario,
                            repeat_id,
                            seed,
                            original,
                            corrupted,
                            statistics,
                        )
                        all_rows.append(row)
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        maybe_save_example(
                            args.output_dir,
                            example_registry,
                            level,
                            corrupted,
                            row,
                        )
        os.replace(temporary_results, results_path)
    finally:
        if temporary_results.exists():
            temporary_results.unlink()

    expected = manifest["sample_count"] * (1 + len(SCENARIOS) * args.repeats)
    if len(all_rows) != expected:
        raise RuntimeError(f"Result count mismatch: {len(all_rows)} != {expected}")
    clean_rows = [row for row in all_rows if row["scenario_name"] == "Clean"]
    if len(clean_rows) != manifest["sample_count"] or not all(
        row["decode"]["verify_success"]
        and row["decode"]["load_success"]
        and row["decode"]["rgb_success"]
        for row in clean_rows
    ):
        raise RuntimeError("One or more shared-preprocessing Clean JPEGs failed decoding")
    params_by_scenario = {"Clean": None, **SCENARIOS}
    summaries = summarize_results(all_rows, params_by_scenario)
    write_summary_csv(args.output_dir / "scenario_summary.csv", summaries)
    aggregate = aggregate_scenarios(all_rows)

    by_scenario: dict[str, Counter] = defaultdict(Counter)
    by_scenario_repeat: dict[str, Counter] = defaultdict(Counter)
    for row in all_rows:
        name = row["scenario_name"]
        exception = row["decode"]["exception_type"] or "none"
        by_scenario[name][exception] += 1
        by_scenario_repeat[f"{name}/repeat_{row['repeat_id']}"][exception] += 1
    exception_report = {
        "by_scenario": {
            key: dict(sorted(value.items())) for key, value in sorted(by_scenario.items())
        },
        "by_scenario_repeat": {
            key: dict(sorted(value.items()))
            for key, value in sorted(by_scenario_repeat.items())
        },
        "all_failures_included_in_denominators": True,
    }
    write_json(args.output_dir / "exception_summary.json", exception_report)
    ready_count = sum(
        row["scenario_name"] != "Clean" and row["rgb_success_rate"] >= 0.3
        for row in aggregate
    )
    conclusion = (
        "model_benchmark_can_proceed"
        if ready_count >= 3
        else "scenario_recalibration_required"
    )
    render_markdown(
        args.output_dir / "RBCS_DECODE_PILOT.md",
        manifest=manifest,
        manifest_file_sha256=manifest_file_sha256,
        scenario_rows=aggregate,
        summaries=summaries,
        exceptions=exception_report,
        examples=example_registry,
        conclusion=conclusion,
        output_dir=args.output_dir,
    )
    write_json(
        args.output_dir / "pilot_summary.json",
        {
            "pilot_version": PILOT_VERSION,
            "manifest_file_sha256": manifest_file_sha256,
            "result_row_count": len(all_rows),
            "scenario_repeat_summary": summaries,
            "scenario_summary": aggregate,
            "ready_or_partial_nonclean_scenario_count": ready_count,
            "conclusion": conclusion,
            "test_split_read": False,
        },
    )
    print(json.dumps({"conclusion": conclusion, "scenario_summary": aggregate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
