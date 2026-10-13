#!/usr/bin/env python3
"""Cache frozen official ByteFormer outputs for the fixed SpatialSense set."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from spacebyte.data import PREPROCESS_VERSION, load_json
from spacebyte.data.spatialsense import render_record_to_jpeg_bytes
from spacebyte.models import ByteFormerTokenAdapter, build_official_byteformer, file_sha256


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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--consistency-samples", type=int, default=4)
    return parser.parse_args()


@torch.no_grad()
def encode(adapter, byte_tensor, device):
    features, mask = adapter(byte_tensor.unsqueeze(0).to(device))
    if not torch.isfinite(features).all():
        raise RuntimeError("ByteFormer cache contains NaN or Inf")
    return features[0].float().cpu(), mask[0].long().cpu()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    manifest = load_json(args.manifest)
    records = list(manifest["samples"])
    byteformer, _, load_report = build_official_byteformer(
        args.config, args.checkpoint
    )
    adapter = ByteFormerTokenAdapter(byteformer).to(device).eval()
    adapter.requires_grad_(False)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    cache_records = []
    online = {}
    for record in records:
        jpeg, _, _ = render_record_to_jpeg_bytes(args.dataset_root, record)
        features, mask = encode(adapter, jpeg, device)
        valid_tokens = int(mask.sum().item())
        cached = {
            "sample_id": record["sample_id"],
            "byte_features": features.half(),
            "byte_attention_mask": mask.to(torch.uint8),
            "valid_token_count": valid_tokens,
            "jpeg_byte_length": jpeg.numel(),
            "feature_shape": list(features.shape),
            "byteformer_checkpoint_sha256": load_report["checkpoint_sha256"],
            "preprocess_version": PREPROCESS_VERSION,
            "cache_dtype": "torch.float16",
        }
        path = args.cache_root / f"{record['sample_id']}.pt"
        torch.save(cached, path)
        online[record["sample_id"]] = (jpeg, features, mask)
        cache_records.append(
            {
                "sample_id": record["sample_id"],
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "jpeg_byte_length": jpeg.numel(),
                "feature_shape": list(features.shape),
                "valid_token_count": valid_tokens,
            }
        )

    selected = list(records)
    random.Random(manifest["seed"] + 1).shuffle(selected)
    consistency = []
    for record in selected[: args.consistency_samples]:
        sample_id = record["sample_id"]
        jpeg, original_features, original_mask = online[sample_id]
        repeated_features, repeated_mask = encode(adapter, jpeg, device)
        stored = torch.load(
            args.cache_root / f"{sample_id}.pt", map_location="cpu", weights_only=True
        )
        restored = stored["byte_features"].float()
        difference = (repeated_features - restored).abs()
        online_repeat_difference = (repeated_features - original_features).abs()
        consistency.append(
            {
                "sample_id": sample_id,
                "max_absolute_error_fp16_cache": float(difference.max().item()),
                "mean_absolute_error_fp16_cache": float(difference.mean().item()),
                "online_repeat_max_absolute_error": float(
                    online_repeat_difference.max().item()
                ),
                "mask_exact": bool(
                    torch.equal(repeated_mask, stored["byte_attention_mask"].long())
                    and torch.equal(repeated_mask, original_mask)
                ),
            }
        )
    report = {
        "manifest": str(args.manifest.resolve()),
        "sample_count": len(records),
        "cache_root": str(args.cache_root.resolve()),
        "cache_dtype": "torch.float16",
        "preprocess_version": PREPROCESS_VERSION,
        "byteformer_config": str(args.config.resolve()),
        "byteformer_config_sha256": file_sha256(args.config),
        "byteformer_checkpoint": str(args.checkpoint.resolve()),
        "byteformer_checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_load": load_report,
        "records": cache_records,
        "total_cache_bytes": sum(row["size_bytes"] for row in cache_records),
        "consistency": consistency,
        "all_masks_exact": all(row["mask_exact"] for row in consistency),
        "max_cache_absolute_error": max(
            row["max_absolute_error_fp16_cache"] for row in consistency
        ),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
    report_path = args.cache_root / "cache_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
