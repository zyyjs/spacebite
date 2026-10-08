#!/usr/bin/env python3
"""Evaluate a fine-tuned SpatialSense ByteFormer checkpoint."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from train import evaluate

from spacebyte.data.spatialsense import SpatialSenseByteDataset, collate_byte_samples
from spacebyte.models import build_spatialsense_byteformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = config["paths"]
    dataset = SpatialSenseByteDataset(Path(paths["processed_root"]), args.split)
    loader = DataLoader(
        dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["workers"]),
        collate_fn=collate_byte_samples,
    )
    model, _ = build_spatialsense_byteformer(
        Path(paths["corenet_config"]), Path(paths["pretrained_checkpoint"])
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device)
    model.to(device)
    loss, metrics = evaluate(model, loader, device, config["training"]["precision"])
    report = {
        "split": args.split,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "samples": len(dataset),
        "loss": loss,
        "metrics": metrics,
    }
    report_text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_text + "\n", encoding="utf-8")
    print(report_text)


if __name__ == "__main__":
    main()
