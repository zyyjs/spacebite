#!/usr/bin/env python3
"""Decompose the previous pilot loss into answer-token and EOS NLL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from spacebyte.answer_only import audit_answer_eos_nll
from spacebyte.data import SpatialSenseFullCachedDataset
from spacebyte.full_pipeline import build_full_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--bridge-checkpoint", type=Path, required=True)
    parser.add_argument("--vicuna-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--qformer-tokenizer",
        type=Path,
        default=Path("/data/students/Zhang_jinsong/spacebyte/models/bert-base-uncased-tokenizer"),
    )
    parser.add_argument(
        "--blip2-checkpoint",
        type=Path,
        default=Path("/data/students/Zhang_jinsong/spacebyte/models/blip2_stage1/blip2_pretrained.pth"),
    )
    parser.add_argument(
        "--byteformer-config",
        type=Path,
        default=Path("/home/Zhang_jinsong/spacebyte/corenet/projects/byteformer/imagenet_jpeg_q100/conv_kernel_size=8.yaml"),
    )
    parser.add_argument(
        "--byteformer-checkpoint",
        type=Path,
        default=Path("/data/students/Zhang_jinsong/spacebyte/models/byteformer_jpeg_q100_k8_w128/imagenet_jpeg_q100_k8_w128.pt"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model, load_report = build_full_model(
        device=device,
        byteformer_config=args.byteformer_config,
        byteformer_checkpoint=args.byteformer_checkpoint,
        blip2_checkpoint=args.blip2_checkpoint,
        vicuna_path=args.vicuna_path,
        qformer_tokenizer_path=args.qformer_tokenizer,
        bridge_checkpoint=args.bridge_checkpoint,
    )
    datasets = {
        split: SpatialSenseFullCachedDataset(
            manifest, args.cache_root, expected_split=split
        )
        for split, manifest in (
            ("train", args.train_manifest),
            ("valid", args.valid_manifest),
        )
    }
    report = {
        "checkpoint": str(args.bridge_checkpoint.resolve()),
        "checkpoint_load": load_report["bridge_checkpoint_load"],
        "splits": {
            split: audit_answer_eos_nll(
                model,
                dataset,
                device=device,
                batch_size=args.batch_size,
                num_workers=2,
            )
            for split, dataset in datasets.items()
        },
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
