#!/usr/bin/env python3
"""Evaluate a bridge checkpoint on SpatialSense validation caches only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from spacebyte.data import SpatialSenseFullCachedDataset
from spacebyte.full_pipeline import build_full_model, evaluate_full


DEFAULT_QFORMER_TOKENIZER = Path(
    "/data/students/Zhang_jinsong/spacebyte/models/bert-base-uncased-tokenizer"
)
DEFAULT_BLIP2 = Path(
    "/data/students/Zhang_jinsong/spacebyte/models/blip2_stage1/blip2_pretrained.pth"
)
DEFAULT_BYTEFORMER_CONFIG = Path(
    "/home/Zhang_jinsong/spacebyte/corenet/projects/byteformer/"
    "imagenet_jpeg_q100/conv_kernel_size=8.yaml"
)
DEFAULT_BYTEFORMER_CHECKPOINT = Path(
    "/data/students/Zhang_jinsong/spacebyte/models/"
    "byteformer_jpeg_q100_k8_w128/imagenet_jpeg_q100_k8_w128.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--vicuna-path", type=Path, required=True)
    parser.add_argument("--bridge-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--qformer-tokenizer", type=Path, default=DEFAULT_QFORMER_TOKENIZER)
    parser.add_argument("--blip2-checkpoint", type=Path, default=DEFAULT_BLIP2)
    parser.add_argument("--byteformer-config", type=Path, default=DEFAULT_BYTEFORMER_CONFIG)
    parser.add_argument("--byteformer-checkpoint", type=Path, default=DEFAULT_BYTEFORMER_CHECKPOINT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Full Vicuna validation requires CUDA")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    dataset = SpatialSenseFullCachedDataset(
        args.manifest, args.cache_root, expected_split="valid"
    )
    model, load_report = build_full_model(
        device=device,
        byteformer_config=args.byteformer_config,
        byteformer_checkpoint=args.byteformer_checkpoint,
        blip2_checkpoint=args.blip2_checkpoint,
        vicuna_path=args.vicuna_path,
        qformer_tokenizer_path=args.qformer_tokenizer,
        bridge_checkpoint=args.bridge_checkpoint,
    )
    evaluation = evaluate_full(
        model, dataset, device=device, batch_size=args.batch_size
    )
    report = {
        "split": "valid",
        "load_report": load_report,
        "evaluation": evaluation,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "split": "valid",
                "greedy": evaluation["greedy"],
                "likelihood": evaluation["likelihood"],
                "peak_gpu_memory_bytes": report["peak_gpu_memory_bytes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
