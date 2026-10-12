#!/usr/bin/env python3
"""Inspect the official BLIP-2 Stage-1 checkpoint against ByteQFormerBridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from spacebyte.models import ByteQFormerBridge, inspect_blip2_qformer_checkpoint


DEFAULT_CHECKPOINT = Path(
    "/data/students/Zhang_jinsong/spacebyte/models/blip2_stage1/" "blip2_pretrained.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(20261012)
    bridge = ByteQFormerBridge()
    report = inspect_blip2_qformer_checkpoint(bridge, args.checkpoint)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
