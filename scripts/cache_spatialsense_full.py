#!/usr/bin/env python3
"""Build a resumable, content-addressed ByteFormer cache for SpatialSense."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from spacebyte.data import (
    PREPROCESS_VERSION,
    VISUAL_PREPROCESS_CONFIG,
    atomic_torch_save,
    flatten_spatialsense_annotations,
    sha256_file,
    validate_feature_cache_file,
    visual_cache_identity,
)
from spacebyte.data.spatialsense import render_record_to_jpeg_bytes
from spacebyte.jpeg_bytes import pad_byte_sequences
from spacebyte.models import ByteFormerTokenAdapter, build_official_byteformer, file_sha256


SPLITS = ("train", "valid", "test")
EXPECTED_SPLIT_COUNTS = {"train": 11238, "valid": 2638, "test": 3622}
DEFAULT_CONFIG = Path(
    "/home/Zhang_jinsong/spacebyte/corenet/projects/byteformer/"
    "imagenet_jpeg_q100/conv_kernel_size=8.yaml"
)
DEFAULT_CHECKPOINT = Path(
    "/data/students/Zhang_jinsong/spacebyte/models/"
    "byteformer_jpeg_q100_k8_w128/imagenet_jpeg_q100_k8_w128.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--consistency-samples-per-split", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261013)
    return parser.parse_args()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def cache_relative_path(key: str) -> str:
    return str(Path("features") / key[:2] / f"{key}.pt")


@torch.no_grad()
def encode_batch(adapter, streams: list[torch.Tensor], device: torch.device):
    byte_batch = pad_byte_sequences(streams).to(device)
    features, masks = adapter(byte_batch)
    if not bool(torch.isfinite(features).all().item()):
        raise RuntimeError("Online ByteFormer output contains NaN or Inf")
    return features.float().cpu(), masks.long().cpu()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.consistency_samples_per_split < 10:
        raise ValueError("batch size must be positive and consistency checks >=10/split")
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    records = flatten_spatialsense_annotations(args.dataset_root)
    split_counts = Counter(row["split"] for row in records)
    if dict(split_counts) != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(f"Unexpected official split counts: {dict(split_counts)}")
    sample_ids = [row["sample_id"] for row in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("SpatialSense sample IDs are not globally unique")

    checkpoint_sha = file_sha256(args.checkpoint)
    raw_sha_by_path = {
        path: sha256_file(args.dataset_root / path)
        for path in sorted({row["image_path"] for row in records})
    }
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identity_by_key: dict[str, dict[str, Any]] = {}
    for record in records:
        key, identity = visual_cache_identity(
            record,
            raw_image_sha256=raw_sha_by_path[record["image_path"]],
            byteformer_checkpoint_sha256=checkpoint_sha,
        )
        record["visual_cache_key"] = key
        record["cache_relative_path"] = cache_relative_path(key)
        groups[key].append(record)
        identity_by_key[key] = identity

    byteformer, _, load_report = build_official_byteformer(
        args.config, args.checkpoint
    )
    adapter = ByteFormerTokenAdapter(byteformer).to(device).eval()
    adapter.requires_grad_(False)
    resumed = 0
    regenerated_invalid = Counter()
    generated = 0
    keys_to_generate = []
    for key in sorted(groups):
        path = args.cache_root / cache_relative_path(key)
        valid, reason = validate_feature_cache_file(
            path, expected_visual_cache_key=key
        )
        if valid:
            resumed += 1
        else:
            if reason != "missing":
                regenerated_invalid[reason] += 1
            keys_to_generate.append(key)

    print(
        f"annotations={len(records)} unique_visuals={len(groups)} "
        f"resumed={resumed} pending={len(keys_to_generate)}",
        flush=True,
    )
    for batch_start in range(0, len(keys_to_generate), args.batch_size):
        batch_keys = keys_to_generate[batch_start : batch_start + args.batch_size]
        streams = []
        rendered_meta = []
        for key in batch_keys:
            record = groups[key][0]
            jpeg, _, boxes = render_record_to_jpeg_bytes(args.dataset_root, record)
            streams.append(jpeg)
            rendered_meta.append(
                {
                    "processed_jpeg_sha256": hashlib.sha256(
                        bytes(jpeg.tolist())
                    ).hexdigest(),
                    "jpeg_byte_length": int(jpeg.numel()),
                    "rendered_boxes": boxes,
                }
            )
        features, masks = encode_batch(adapter, streams, device)
        for index, key in enumerate(batch_keys):
            valid_count = int(masks[index].sum().item())
            if not bool(masks[index, :valid_count].all().item()) or bool(
                masks[index, valid_count:].any().item()
            ):
                raise RuntimeError(f"Non-contiguous ByteFormer mask for {key}")
            member_rows = groups[key]
            cached = {
                "sample_id": member_rows[0]["sample_id"],
                "sample_ids": [row["sample_id"] for row in member_rows],
                "split": member_rows[0]["split"],
                "splits": sorted({row["split"] for row in member_rows}),
                "visual_cache_key": key,
                "visual_identity": identity_by_key[key],
                "byte_features": features[index, :valid_count].half().contiguous(),
                "byte_attention_mask": masks[index, :valid_count]
                .to(torch.uint8)
                .contiguous(),
                "valid_token_count": valid_count,
                **rendered_meta[index],
                "preprocessing_version": PREPROCESS_VERSION,
                "preprocessing": VISUAL_PREPROCESS_CONFIG,
                "byteformer_checkpoint_sha256": checkpoint_sha,
                "cache_dtype": "torch.float16",
            }
            atomic_torch_save(
                cached, args.cache_root / cache_relative_path(key)
            )
            generated += 1
        completed = min(batch_start + len(batch_keys), len(keys_to_generate))
        if completed % 100 == 0 or completed == len(keys_to_generate):
            elapsed = time.perf_counter() - started
            print(
                f"generated {completed}/{len(keys_to_generate)} pending visuals "
                f"({completed / max(elapsed, 1e-6):.2f}/s)",
                flush=True,
            )

    manifest_paths = {}
    manifest_sha = {}
    for split in SPLITS:
        rows = []
        for record in records:
            if record["split"] != split:
                continue
            rows.append(
                {
                    key: record[key]
                    for key in (
                        "sample_id",
                        "split",
                        "visual_cache_key",
                        "cache_relative_path",
                        "image_path",
                        "subject",
                        "object",
                        "relation_raw",
                        "relation",
                        "label",
                        "answer",
                        "question",
                    )
                }
            )
        if len(rows) != EXPECTED_SPLIT_COUNTS[split]:
            raise RuntimeError(f"Manifest count mismatch for {split}")
        path = args.cache_root / "manifests" / f"{split}.jsonl"
        atomic_write_text(
            path,
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        )
        manifest_paths[split] = str(path.resolve())
        manifest_sha[split] = sha256_file(path)

    validation_counts = Counter()
    validation_errors = Counter()
    for key in groups:
        path = args.cache_root / cache_relative_path(key)
        valid, reason = validate_feature_cache_file(
            path, expected_visual_cache_key=key
        )
        validation_counts["valid" if valid else "invalid"] += 1
        if not valid:
            validation_errors[reason] += 1
    if validation_counts["invalid"]:
        raise RuntimeError(f"Invalid cache files after generation: {validation_errors}")

    rng = random.Random(args.seed)
    consistency = []
    for split in SPLITS:
        candidates = [row for row in records if row["split"] == split]
        rng.shuffle(candidates)
        for record in candidates[: args.consistency_samples_per_split]:
            jpeg, _, _ = render_record_to_jpeg_bytes(args.dataset_root, record)
            online_features, online_mask = encode_batch(adapter, [jpeg], device)
            cached = torch.load(
                args.cache_root / record["cache_relative_path"],
                map_location="cpu",
                weights_only=True,
            )
            restored = cached["byte_features"].float()
            online_valid = int(online_mask[0].sum().item())
            if online_valid != restored.shape[0]:
                raise RuntimeError("Online/cache valid token count mismatch")
            difference = (online_features[0, :online_valid] - restored).abs()
            consistency.append(
                {
                    "split": split,
                    "sample_id": record["sample_id"],
                    "visual_cache_key": record["visual_cache_key"],
                    "max_absolute_error": float(difference.max().item()),
                    "mean_absolute_error": float(difference.mean().item()),
                    "mask_exact": bool(
                        torch.equal(
                            online_mask[0, :online_valid],
                            cached["byte_attention_mask"].long(),
                        )
                    ),
                }
            )
    if not all(row["mask_exact"] for row in consistency):
        raise RuntimeError("Online/cache mask consistency failed")

    cache_files = [
        args.cache_root / cache_relative_path(key) for key in groups
    ]
    report = {
        "task": "Byte-stream-conditioned spatial relation verification",
        "annotation_count": len(records),
        "split_annotation_counts": dict(split_counts),
        "unique_sample_id_count": len(set(sample_ids)),
        "unique_visual_cache_count": len(groups),
        "deduplicated_annotation_count": len(records) - len(groups),
        "cache_root": str(args.cache_root.resolve()),
        "cache_size_bytes": sum(path.stat().st_size for path in cache_files),
        "generated_visuals_this_run": generated,
        "resumed_visuals_this_run": resumed,
        "regenerated_invalid": dict(regenerated_invalid),
        "manifest_paths": manifest_paths,
        "manifest_sha256": manifest_sha,
        "byteformer_checkpoint": str(args.checkpoint.resolve()),
        "byteformer_checkpoint_sha256": checkpoint_sha,
        "byteformer_config": str(args.config.resolve()),
        "byteformer_config_sha256": file_sha256(args.config),
        "checkpoint_load": load_report,
        "preprocessing": VISUAL_PREPROCESS_CONFIG,
        "cache_dtype": "torch.float16",
        "cache_validation": {
            "counts": dict(validation_counts),
            "errors": dict(validation_errors),
        },
        "consistency_samples_per_split": args.consistency_samples_per_split,
        "consistency": consistency,
        "max_cache_absolute_error": max(
            row["max_absolute_error"] for row in consistency
        ),
        "mean_cache_absolute_error": sum(
            row["mean_absolute_error"] for row in consistency
        )
        / len(consistency),
        "all_masks_exact": all(row["mask_exact"] for row in consistency),
        "duration_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
    }
    report_path = args.cache_root / "cache_report.json"
    atomic_write_text(
        report_path,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
