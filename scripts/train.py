#!/usr/bin/env python3
"""Train ByteFormer on sparsely annotated SpatialSense relations."""

from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from spacebyte.data.spatialsense import (SpatialSenseByteDataset,
                                         build_weighted_sampler,
                                         collate_byte_samples)
from spacebyte.losses import masked_binary_cross_entropy
from spacebyte.metrics import compute_metrics
from spacebyte.models import build_spatialsense_byteformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--head-warmup-epochs", type=int)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_backbone_trainable(model: torch.nn.Module, trainable: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    precision: str,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    losses = []
    logits_all = []
    targets_all = []
    masks_all = []
    for batch in loader:
        samples = batch["samples"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        with autocast_context(device, precision):
            logits = model(samples)
            loss = masked_binary_cross_entropy(logits, targets, masks)
        losses.append(float(loss))
        logits_all.append(logits.float().cpu())
        targets_all.append(targets.cpu())
        masks_all.append(masks.cpu())
    metrics = compute_metrics(
        torch.cat(logits_all), torch.cat(targets_all), torch.cat(masks_all)
    )
    return float(np.mean(losses)), metrics


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = config["paths"]
    data_config = config["data"]
    train_config = config["training"]
    set_seed(int(train_config["seed"]))

    requested_device = args.device or train_config["device"]
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    smoke_limit_train = 16 if args.smoke else None
    smoke_limit_valid = 8 if args.smoke else None
    processed_root = args.processed_root or Path(paths["processed_root"])
    train_dataset = SpatialSenseByteDataset(
        processed_root, "train", limit=smoke_limit_train
    )
    valid_dataset = SpatialSenseByteDataset(
        processed_root, "valid", limit=smoke_limit_valid
    )
    batch_size = 2 if args.smoke else int(data_config["batch_size"])
    workers = 0 if args.smoke else int(data_config["workers"])
    sampler = (
        build_weighted_sampler(train_dataset)
        if data_config.get("weighted_sampling", True)
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_byte_samples,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_byte_samples,
    )

    model, _ = build_spatialsense_byteformer(
        Path(paths["corenet_config"]), Path(paths["pretrained_checkpoint"])
    )
    model.to(device)
    epochs = args.epochs or (1 if args.smoke else int(train_config["epochs"]))
    warmup_epochs = (
        args.head_warmup_epochs
        if args.head_warmup_epochs is not None
        else (1 if args.smoke else int(train_config["head_warmup_epochs"]))
    )
    set_backbone_trainable(model, trainable=warmup_epochs == 0)

    backbone_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("classifier.")
    ]
    head_parameters = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": float(train_config["backbone_lr"])},
            {
                "params": head_parameters,
                "lr": float(train_config["warmup_head_lr"]),
            },
        ],
        weight_decay=float(train_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    use_scaler = device.type == "cuda" and train_config["precision"] == "fp16"
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    run_name = args.run_name or (
        ("smoke_" if args.smoke else "train_")
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir = Path(paths["result_root"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    log_path = output_dir / "metrics.jsonl"
    best_score = -math.inf

    for epoch in range(epochs):
        if epoch == warmup_epochs:
            set_backbone_trainable(model, trainable=True)
            optimizer.param_groups[1]["lr"] = float(train_config["head_lr"])
        model.train()
        running_losses = []
        for batch in train_loader:
            samples = batch["samples"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            masks = batch["masks"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, train_config["precision"]):
                logits = model(samples)
                loss = masked_binary_cross_entropy(logits, targets, masks)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(train_config["gradient_clip"])
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(train_config["gradient_clip"])
                )
                optimizer.step()
            running_losses.append(float(loss.detach()))

        validation_loss, metrics = evaluate(
            model, valid_loader, device, train_config["precision"]
        )
        scheduler.step()
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(running_losses)),
            "validation_loss": validation_loss,
            "metrics": metrics,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")
        print(json.dumps(epoch_record, ensure_ascii=False))

        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "metrics": metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        score = metrics["macro"].get("balanced_accuracy")
        if score is not None and score > best_score:
            best_score = score
            torch.save(checkpoint, output_dir / "best.pt")

    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
