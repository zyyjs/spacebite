#!/usr/bin/env python3
"""Prepare pair-marked JPEG Q100 samples for ByteFormer."""

import argparse
import json
from pathlib import Path

from spacebyte.data.spatialsense import prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-size", type=int, default=224)
    parser.add_argument("--jpeg-quality", type=int, default=100)
    parser.add_argument("--limit-per-split", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = prepare_dataset(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        output_size=args.output_size,
        jpeg_quality=args.jpeg_quality,
        limit_per_split=args.limit_per_split,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
