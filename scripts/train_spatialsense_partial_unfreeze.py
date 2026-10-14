#!/usr/bin/env python3
"""Train the fixed three-epoch SpatialSense partial-ByteFormer experiment."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from spacebyte.answer_only import audit_answer_only_labels
from spacebyte.data import atomic_torch_save, sha256_file
from spacebyte.evaluation import fixed_derangement
from spacebyte.full_pipeline import build_full_model
from spacebyte.partial_unfreeze import (
    PermutedOnlineByteDataset,
    SpatialSenseOnlineByteDataset,
    build_partial_unfreeze_optimizer_groups,
    collate_online_bytes,
    configure_byteformer_partial_unfreeze,
    evaluate_online_answer_only,
    set_partial_training_mode,
)
from spacebyte.training import gradient_norm_for_names, warmup_cosine_multiplier


ROOT = Path("/home/Zhang_jinsong/spacebyte")
DEFAULT_BRIDGE = Path(
    "/data/students/Zhang_jinsong/spacebyte/result/byteformer/"
    "spatialsense_answer_loss_pilot/train/bridge_best.pt"
)
DEFAULT_QFORMER_TOKENIZER = Path(
    "/data/students/Zhang_jinsong/spacebyte/models/bert-base-uncased-tokenizer"
)
DEFAULT_BLIP2 = Path(
    "/data/students/Zhang_jinsong/spacebyte/models/blip2_stage1/blip2_pretrained.pth"
)
DEFAULT_BYTEFORMER_CONFIG = ROOT / (
    "corenet/projects/byteformer/imagenet_jpeg_q100/conv_kernel_size=8.yaml"
)
DEFAULT_BYTEFORMER_CHECKPOINT = Path(
    "/data/students/Zhang_jinsong/spacebyte/models/"
    "byteformer_jpeg_q100_k8_w128/imagenet_jpeg_q100_k8_w128.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--vicuna-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bridge-checkpoint", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--byteformer-lr", type=float, default=1e-6)
    parser.add_argument("--pretrained-lr", type=float, default=5e-6)
    parser.add_argument("--new-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clipping", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-scaler-init", type=float, default=1024.0)
    parser.add_argument("--smoke-optimizer-steps", type=int, default=20)
    parser.add_argument("--qformer-tokenizer", type=Path, default=DEFAULT_QFORMER_TOKENIZER)
    parser.add_argument("--blip2-checkpoint", type=Path, default=DEFAULT_BLIP2)
    parser.add_argument("--byteformer-config", type=Path, default=DEFAULT_BYTEFORMER_CONFIG)
    parser.add_argument("--byteformer-checkpoint", type=Path, default=DEFAULT_BYTEFORMER_CHECKPOINT)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git_metadata() -> dict[str, str]:
    def run(*arguments: str) -> str:
        return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_short": run("status", "--short"),
    }


def build_model(args: argparse.Namespace, device: torch.device):
    model, load_report = build_full_model(
        device=device,
        byteformer_config=args.byteformer_config,
        byteformer_checkpoint=args.byteformer_checkpoint,
        blip2_checkpoint=args.blip2_checkpoint,
        vicuna_path=args.vicuna_path,
        qformer_tokenizer_path=args.qformer_tokenizer,
        bridge_checkpoint=args.bridge_checkpoint,
    )
    bridge_load = load_report.get("bridge_checkpoint_load")
    if (
        bridge_load is None
        or not bridge_load.get("strict")
        or bridge_load.get("missing_keys")
        or bridge_load.get("unexpected_keys")
    ):
        raise RuntimeError(f"Answer-only bridge checkpoint load failed: {bridge_load}")
    unfreeze_report = configure_byteformer_partial_unfreeze(model)
    groups, optimizer_report = build_partial_unfreeze_optimizer_groups(
        model,
        byteformer_lr=args.byteformer_lr,
        pretrained_lr=args.pretrained_lr,
        new_lr=args.new_lr,
        weight_decay=args.weight_decay,
    )
    return model, load_report, unfreeze_report, groups, optimizer_report


def named_trainable(model) -> dict[str, torch.nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def optimizer_gradient_norms(optimizer) -> dict[str, float]:
    result = {}
    for group in optimizer.param_groups:
        squared = 0.0
        for parameter in group["params"]:
            if parameter.grad is not None:
                squared += float(parameter.grad.detach().float().pow(2).sum().item())
        result[group["group_name"]] = squared**0.5
    return result


def train_micro_batches(
    *,
    model,
    loader,
    optimizer,
    scaler,
    accumulation: int,
    clipping: float,
    target_optimizer_steps: int,
    scheduler=None,
    log_handle=None,
    starting_optimizer_step: int = 0,
    epoch: int | None = None,
) -> dict[str, Any]:
    set_partial_training_mode(model, True)
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    optimizer_step = starting_optimizer_step
    local_optimizer_steps = 0
    loss_sum = 0.0
    loss_count = 0
    nonfinite_events = 0
    gradient_samples = []
    started = time.perf_counter()
    total_micro_batches = len(loader)
    for batch in loader:
        group_start = (micro_step // accumulation) * accumulation
        group_size = min(accumulation, total_micro_batches - group_start)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(
                batch["byte_tensor"].to(model.device, non_blocking=True),
                batch["questions"],
                batch["answers"],
                answer_only_loss=True,
            )
            if not bool(torch.isfinite(output.loss).item()):
                nonfinite_events += 1
                raise RuntimeError("Non-finite answer-only loss")
            loss = output.loss / group_size
        scaler.scale(loss).backward()
        loss_sum += float(output.loss.detach().float().item())
        loss_count += 1
        micro_step += 1
        group_complete = micro_step % accumulation == 0 or micro_step == total_micro_batches
        if not group_complete:
            continue

        scaler.unscale_(optimizer)
        group_gradients = optimizer_gradient_norms(optimizer)
        total_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping).item()
        )
        if not math.isfinite(total_norm) or not all(
            math.isfinite(value) and value > 0 for value in group_gradients.values()
        ):
            nonfinite_events += 1
            raise RuntimeError(
                f"Invalid gradient at local step {local_optimizer_steps + 1}: "
                f"total={total_norm}, groups={group_gradients}"
            )
        learning_rates = {
            group["group_name"]: group["lr"] for group in optimizer.param_groups
        }
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
        local_optimizer_steps += 1
        gradient_samples.append(group_gradients)
        if log_handle is not None:
            row = {
                "epoch": epoch,
                "optimizer_step": optimizer_step,
                "epoch_optimizer_step": local_optimizer_steps,
                "micro_step": micro_step,
                "mean_loss": loss_sum / loss_count,
                "learning_rates_used": learning_rates,
                "gradient_norms": group_gradients,
                "total_gradient_norm_before_clipping": total_norm,
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(model.device)),
                "elapsed_seconds": time.perf_counter() - started,
                "nonfinite_events": nonfinite_events,
            }
            log_handle.write(json.dumps(row) + "\n")
            log_handle.flush()
        loss_sum = 0.0
        loss_count = 0
        if local_optimizer_steps == target_optimizer_steps:
            break
    if local_optimizer_steps != target_optimizer_steps:
        raise RuntimeError(
            f"Expected {target_optimizer_steps} optimizer steps, got {local_optimizer_steps}"
        )
    return {
        "optimizer_steps": local_optimizer_steps,
        "micro_steps": micro_step,
        "duration_seconds": time.perf_counter() - started,
        "nonfinite_events": nonfinite_events,
        "gradient_norm_min_by_group": {
            name: min(row[name] for row in gradient_samples)
            for name in gradient_samples[0]
        },
        "gradient_norm_max_by_group": {
            name: max(row[name] for row in gradient_samples)
            for name in gradient_samples[0]
        },
    }


def run_smoke(
    args: argparse.Namespace,
    device: torch.device,
    train_dataset: SpatialSenseOnlineByteDataset,
) -> dict[str, Any]:
    set_seed(args.seed)
    model, load_report, unfreeze, groups, optimizer_report = build_model(args, device)
    optimizer = torch.optim.AdamW(groups)
    scaler = torch.cuda.amp.GradScaler(
        init_scale=args.grad_scaler_init, growth_interval=2000
    )
    trainable = named_trainable(model)
    before_trainable = {
        name: parameter.detach().cpu().clone()
        for name, parameter in trainable.items()
    }
    frozen_versions = {
        name: parameter._version
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        collate_fn=collate_online_bytes,
        pin_memory=args.num_workers > 0,
    )
    torch.cuda.reset_peak_memory_stats(device)
    training = train_micro_batches(
        model=model,
        loader=loader,
        optimizer=optimizer,
        scaler=scaler,
        accumulation=args.gradient_accumulation,
        clipping=args.gradient_clipping,
        target_optimizer_steps=args.smoke_optimizer_steps,
    )
    changed = {
        name: not torch.equal(before_trainable[name], parameter.detach().cpu())
        for name, parameter in trainable.items()
    }
    changes_by_group = {}
    optimizer_name_groups = optimizer_report["groups"]
    for group_name, group in optimizer_name_groups.items():
        names = [name.removeprefix("byteformer.").removeprefix("bridge.") for name in group["parameter_names"]]
        prefix = "byteformer_adapter.byteformer." if group_name == "byteformer_unfrozen" else "bridge."
        full_names = [prefix + name for name in names]
        changes_by_group[group_name] = {
            "tensor_count": len(full_names),
            "changed_tensor_count": sum(changed[name] for name in full_names),
            "any_parameter_updated": any(changed[name] for name in full_names),
        }
    frozen_version_changes = [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad and parameter._version != frozen_versions[name]
    ]
    frozen_with_grad = [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    if frozen_version_changes or frozen_with_grad:
        raise RuntimeError(
            f"Frozen parameter smoke audit failed: versions={frozen_version_changes}, "
            f"grads={frozen_with_grad}"
        )
    if not all(row["any_parameter_updated"] for row in changes_by_group.values()):
        raise RuntimeError(f"A trainable group did not update: {changes_by_group}")
    report = {
        "passed": True,
        "fresh_reload_after_smoke_required": True,
        "load_report": load_report,
        "unfreeze": unfreeze,
        "optimizer": optimizer_report,
        "training": training,
        "trainable_group_updates": changes_by_group,
        "frozen_parameter_tensor_count": len(frozen_versions),
        "frozen_parameter_versions_unchanged": True,
        "frozen_parameters_with_grad": [],
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    del model, optimizer, scaler, before_trainable, trainable
    gc.collect()
    torch.cuda.empty_cache()
    return report


def checkpoint_artifact(model, *, epoch: int, metadata: dict[str, Any]):
    trainable_names = {
        name
        for name, parameter in model.byteformer_adapter.byteformer.named_parameters()
        if parameter.requires_grad
    }
    byteformer_state = {
        name: value.detach().cpu().clone()
        for name, value in model.byteformer_adapter.byteformer.state_dict().items()
        if name in trainable_names
    }
    return {
        "format_version": 1,
        "bridge_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.bridge.state_dict().items()
        },
        "byteformer_partial_state_dict": byteformer_state,
        "epoch": epoch,
        "metadata": metadata,
    }


def metric_summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in evaluation.items() if key != "outputs"}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("The fixed FP16 experiment requires CUDA")
    if args.gradient_accumulation != 8 or args.smoke_optimizer_steps != 20:
        raise ValueError("This experiment fixes accumulation=8 and smoke steps=20")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    train_dataset = SpatialSenseOnlineByteDataset(
        args.dataset_root, args.train_manifest, expected_split="train"
    )
    valid_dataset = SpatialSenseOnlineByteDataset(
        args.dataset_root, args.valid_manifest, expected_split="valid"
    )
    if (len(train_dataset), len(valid_dataset)) != (11238, 2638):
        raise RuntimeError("Official train/valid counts changed")
    if set(row["sample_id"] for row in train_dataset.records) & set(
        row["sample_id"] for row in valid_dataset.records
    ):
        raise RuntimeError("Train/valid sample IDs overlap")

    smoke = run_smoke(args, device, train_dataset)
    write_json(args.output_dir / "smoke_20_steps.json", smoke)

    # The smoke optimizer state is deliberately discarded. Formal training is
    # rebuilt from the unchanged official ByteFormer plus answer-only bridge.
    set_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    model, load_report, unfreeze, groups, optimizer_report = build_model(args, device)
    optimizer = torch.optim.AdamW(groups)
    steps_per_epoch = math.ceil(len(train_dataset) / args.gradient_accumulation)
    if steps_per_epoch != 1405:
        raise RuntimeError(f"Expected 1405 steps/epoch, got {steps_per_epoch}")
    total_optimizer_steps = 3 * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: warmup_cosine_multiplier(
            step, warmup_steps=args.warmup_steps, total_steps=total_optimizer_steps
        ),
    )
    scaler = torch.cuda.amp.GradScaler(
        init_scale=args.grad_scaler_init, growth_interval=2000
    )

    audit_batch = collate_online_bytes([train_dataset[0], train_dataset[1]])
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        audit_output = model(
            audit_batch["byte_tensor"].to(device),
            audit_batch["questions"],
            audit_batch["answers"],
            answer_only_loss=True,
        )
    label_audit = audit_answer_only_labels(
        audit_output.prepared, audit_batch["answers"], model.llm_tokenizer
    )
    if not label_audit["exactly_one_supervised_token_per_sample"] or not label_audit["eos_fully_ignored"]:
        raise RuntimeError("Answer-only label audit failed")

    visual_keys = [row["visual_identity_key"] for row in valid_dataset.records]
    permutation = fixed_derangement(len(valid_dataset), args.seed, visual_keys)
    shuffled_dataset = PermutedOnlineByteDataset(valid_dataset, permutation)
    write_json(
        args.output_dir / "valid_shuffled_byte_mapping.json",
        {
            "seed": args.seed,
            "mapping": [
                {
                    "target_index": index,
                    "source_index": source,
                    "target_sample_id": valid_dataset.records[index]["sample_id"],
                    "source_sample_id": valid_dataset.records[source]["sample_id"],
                }
                for index, source in enumerate(permutation)
            ],
        },
    )
    config = {
        "task": "SpatialSense answer-only partial ByteFormer unfreeze",
        "epochs": 3,
        "feature_cache_used": False,
        "online_input": "red/blue marked PIL JPEG quality 100 byte stream",
        "test_split_used": False,
        "micro_batch_size": 1,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": args.gradient_accumulation,
        "precision": "FP16 autocast with GradScaler",
        "seed": args.seed,
        "learning_rates": {
            "byteformer_unfrozen": args.byteformer_lr,
            "blip2_initialized": args.pretrained_lr,
            "new_modules": args.new_lr,
        },
        "gradient_clipping": args.gradient_clipping,
        "warmup_steps": args.warmup_steps,
        "scheduler": "linear warmup then cosine decay over three epochs",
        "bridge_checkpoint": str(args.bridge_checkpoint.resolve()),
        "bridge_checkpoint_sha256": sha256_file(args.bridge_checkpoint),
        "manifest_sha256": {
            "train": sha256_file(args.train_manifest),
            "valid": sha256_file(args.valid_manifest),
        },
        "unfreeze": unfreeze,
        "optimizer": optimizer_report,
        "model_load": load_report,
        "label_audit": label_audit,
        "smoke_report": str((args.output_dir / "smoke_20_steps.json").resolve()),
        "git_at_launch": git_metadata(),
    }
    write_json(args.output_dir / "training_config.json", config)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        collate_fn=collate_online_bytes,
        pin_memory=args.num_workers > 0,
    )
    curve_path = args.output_dir / "training_curve.jsonl"
    curve_path.write_text("", encoding="utf-8")
    epochs = []
    best_epoch = None
    best_key = None
    global_optimizer_step = 0
    training_seconds = 0.0
    checkpoint_metadata = {"config": config, "smoke": smoke}
    with curve_path.open("a", encoding="utf-8") as log_handle:
        for epoch in range(1, 4):
            training = train_micro_batches(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                accumulation=args.gradient_accumulation,
                clipping=args.gradient_clipping,
                target_optimizer_steps=steps_per_epoch,
                scheduler=scheduler,
                log_handle=log_handle,
                starting_optimizer_step=global_optimizer_step,
                epoch=epoch,
            )
            global_optimizer_step += steps_per_epoch
            training_seconds += training["duration_seconds"]
            evaluations = {}
            for condition, dataset, scale in (
                ("normal", valid_dataset, 1.0),
                ("zero_byte", valid_dataset, 0.0),
                ("shuffled_byte", shuffled_dataset, 1.0),
            ):
                evaluation = evaluate_online_answer_only(
                    model,
                    dataset,
                    device=device,
                    batch_size=args.eval_batch_size,
                    projected_query_scale=scale,
                    num_workers=args.num_workers,
                )
                write_json(
                    args.output_dir / f"epoch_{epoch}_{condition}.json", evaluation
                )
                evaluations[condition] = metric_summary(evaluation)
            normal = evaluations["normal"]["logit_yes_minus_no"]
            control_aurocs = [
                evaluations[name]["logit_yes_minus_no"]["auroc"]
                for name in ("zero_byte", "shuffled_byte")
            ]
            margin = normal["auroc"] - max(control_aurocs)
            epoch_report = {
                "epoch": epoch,
                "training": training,
                "evaluations": evaluations,
                "normal_auroc_margin_over_stronger_control": margin,
                "normal_exceeds_both_controls_by_0_02": margin >= 0.02,
            }
            epochs.append(epoch_report)
            write_json(args.output_dir / f"epoch_{epoch}_summary.json", epoch_report)
            key = (normal["auroc"], normal["accuracy"], normal["macro_f1"])
            artifact = checkpoint_artifact(
                model,
                epoch=epoch,
                metadata={**checkpoint_metadata, "epoch_metrics": epoch_report},
            )
            atomic_torch_save(artifact, args.output_dir / "partial_last.pt")
            if best_key is None or key > best_key:
                best_key = key
                best_epoch = epoch
                atomic_torch_save(artifact, args.output_dir / "partial_best.pt")

    allowed_byteformer_prefixes = tuple(
        f"{name}." for name in unfreeze["unfrozen_modules"]
    )
    frozen_audit = {
        "vicuna_trainable_parameters": sum(
            parameter.numel() for parameter in model.vicuna.parameters() if parameter.requires_grad
        ),
        "vicuna_parameters_with_grad": sum(
            parameter.grad is not None for parameter in model.vicuna.parameters()
        ),
        "byteformer_unexpected_trainable_parameters": [
            name
            for name, parameter in model.byteformer_adapter.byteformer.named_parameters()
            if parameter.requires_grad
            and not name.startswith(allowed_byteformer_prefixes)
        ],
    }
    if any(frozen_audit.values()):
        raise RuntimeError(f"Final frozen-module audit failed: {frozen_audit}")
    report = {
        "config": config,
        "smoke": smoke,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_selection_key": best_key,
        "normal_sustained_0_02_margin": all(
            row["normal_exceeds_both_controls_by_0_02"] for row in epochs
        ),
        "global_optimizer_steps": global_optimizer_step,
        "training_duration_seconds": training_seconds,
        "total_duration_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "frozen_audit": frozen_audit,
        "checkpoints": {
            "best": str((args.output_dir / "partial_best.pt").resolve()),
            "last": str((args.output_dir / "partial_last.pt").resolve()),
        },
        "test_split_used": False,
    }
    write_json(args.output_dir / "training_report.json", report)
    print(json.dumps({"status": "complete", "report": str((args.output_dir / "training_report.json").resolve())}))


if __name__ == "__main__":
    main()
