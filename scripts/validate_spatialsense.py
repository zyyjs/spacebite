#!/usr/bin/env python3
"""Validate a processed SpatialSense ByteFormer dataset."""

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from spacebyte.constants import RELATIONS
from spacebyte.data.spatialsense import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conv-kernel-size", type=int, default=8)
    parser.add_argument("--max-num-tokens", type=int, default=50000)
    args = parser.parse_args()

    metadata = json.loads(
        (args.processed_root / "metadata.json").read_text(encoding="utf-8")
    )
    split_counts = {}
    relation_labels = {name: Counter() for name in RELATIONS}
    sample_ids = set()
    byte_lengths = []
    errors = []
    observed_labels = 0

    for split in ("train", "valid", "test"):
        records = load_manifest(args.processed_root / "manifests" / f"{split}.jsonl")
        split_counts[split] = len(records)
        for record in records:
            sample_id = f"{split}/{record['sample_id']}"
            if sample_id in sample_ids:
                errors.append(f"duplicate sample id: {sample_id}")
            sample_ids.add(sample_id)
            image_path = args.processed_root / record["image_path"]
            if not image_path.is_file():
                errors.append(f"missing image: {image_path}")
                continue
            actual_length = image_path.stat().st_size
            byte_lengths.append(actual_length)
            if actual_length != record["byte_length"]:
                errors.append(f"byte length mismatch: {image_path}")
            expected_size = int(metadata["output_size"])
            with Image.open(image_path) as image:
                if image.format != "JPEG" or image.size != (
                    expected_size,
                    expected_size,
                ):
                    errors.append(
                        f"invalid image encoding/size: {image_path}: {image.format} {image.size}"
                    )
            if len(record["target"]) != len(RELATIONS) or len(record["mask"]) != len(
                RELATIONS
            ):
                errors.append(f"invalid label dimensions: {sample_id}")
                continue
            if sum(record["mask"]) < 1:
                errors.append(f"empty annotation mask: {sample_id}")
            for index, relation in enumerate(RELATIONS):
                if record["mask"][index]:
                    observed_labels += 1
                    label = "positive" if record["target"][index] else "negative"
                    relation_labels[relation][label] += 1

    stride = args.conv_kernel_size // 2
    max_bytes = max(byte_lengths)
    max_reduced_tokens = (max_bytes - args.conv_kernel_size) // stride + 1
    if max_reduced_tokens > args.max_num_tokens:
        errors.append(
            f"maximum reduced token length {max_reduced_tokens} exceeds {args.max_num_tokens}"
        )
    if split_counts != metadata["split_pair_counts"]:
        errors.append("manifest split counts do not match metadata")
    report = {
        "status": "passed" if not errors else "failed",
        "split_pair_counts": split_counts,
        "metadata_split_pair_counts": metadata["split_pair_counts"],
        "total_pairs": sum(split_counts.values()),
        "observed_labels": observed_labels,
        "relation_counts": {
            name: dict(counts) for name, counts in relation_labels.items()
        },
        "byte_length": {
            "min": min(byte_lengths),
            "max": max_bytes,
            "mean": sum(byte_lengths) / len(byte_lengths),
        },
        "maximum_reduced_tokens": max_reduced_tokens,
        "max_num_tokens": args.max_num_tokens,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
