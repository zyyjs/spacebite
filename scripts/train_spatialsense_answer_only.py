#!/usr/bin/env python3
"""Run the fixed one-epoch SpatialSense answer-only loss pilot."""

from __future__ import annotations

import argparse
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

from spacebyte.answer_only import (
    assert_clean_blip2_initialization,
    audit_answer_only_labels,
    evaluate_answer_only,
)
from spacebyte.data import (
    SpatialSenseFullCachedDataset,
    atomic_torch_save,
    collate_cached_features,
    sha256_file,
)
from spacebyte.evaluation import fixed_derangement
from spacebyte.full_pipeline import PermutedVisualDataset, build_full_model
from spacebyte.training import (
    assert_frozen_modules_absent,
    build_bridge_optimizer_groups,
    checkpoint_is_better,
    gradient_norm_for_names,
    warmup_cosine_multiplier,
)


ROOT = Path("/home/Zhang_jinsong/spacebyte")
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
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--vicuna-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--pretrained-lr", type=float, default=1e-5)
    parser.add_argument("--new-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clipping", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-scaler-init", type=float, default=1024.0)
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
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_short": run("status", "--short"),
    }


def bridge_artifact(model, step: int, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": 1,
        "bridge_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.bridge.state_dict().items()
        },
        "optimizer_step": step,
        "metadata": metadata,
    }


def summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in evaluation.items() if key != "outputs"}


def gradient_report(bridge) -> dict[str, float]:
    return {
        "cross_attention_kv": gradient_norm_for_names(
            bridge,
            lambda name: ".crossattention.self.key.weight" in name
            or ".crossattention.self.value.weight" in name,
        ),
        "projector": gradient_norm_for_names(
            bridge, lambda name: name.startswith("llm_projector.")
        ),
        "query_tokens": gradient_norm_for_names(
            bridge, lambda name: name == "query_tokens"
        ),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("This fixed FP16 pilot requires CUDA")
    if args.gradient_accumulation != 8:
        raise ValueError("This comparison fixes gradient accumulation at 8")
    set_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = SpatialSenseFullCachedDataset(
        args.train_manifest, args.cache_root, expected_split="train"
    )
    valid_dataset = SpatialSenseFullCachedDataset(
        args.valid_manifest, args.cache_root, expected_split="valid"
    )
    if (len(train_dataset), len(valid_dataset)) != (11238, 2638):
        raise RuntimeError("Official train/valid annotation counts changed")

    model, load_report = build_full_model(
        device=device,
        byteformer_config=args.byteformer_config,
        byteformer_checkpoint=args.byteformer_checkpoint,
        blip2_checkpoint=args.blip2_checkpoint,
        vicuna_path=args.vicuna_path,
        qformer_tokenizer_path=args.qformer_tokenizer,
        bridge_checkpoint=None,
    )
    assert_clean_blip2_initialization(load_report)
    memory_after_model_load = int(torch.cuda.memory_allocated(device))
    optimizer_groups, optimizer_report = build_bridge_optimizer_groups(
        model.bridge,
        pretrained_lr=args.pretrained_lr,
        new_lr=args.new_lr,
        weight_decay=args.weight_decay,
    )
    assert_frozen_modules_absent(optimizer_groups, model.byteformer_adapter, model.vicuna)
    optimizer = torch.optim.AdamW(optimizer_groups)
    optimizer_steps = math.ceil(len(train_dataset) / args.gradient_accumulation)
    if optimizer_steps != 1405:
        raise RuntimeError(f"Expected 1405 optimizer steps, got {optimizer_steps}")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: warmup_cosine_multiplier(
            step, warmup_steps=args.warmup_steps, total_steps=optimizer_steps
        ),
    )
    scaler = torch.cuda.amp.GradScaler(
        init_scale=args.grad_scaler_init, growth_interval=2000
    )

    audit_batch = next(
        iter(
            DataLoader(
                train_dataset,
                batch_size=2,
                shuffle=False,
                collate_fn=collate_cached_features,
            )
        )
    )
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        audit_output = model.forward_from_byte_features(
            audit_batch["byte_features"].to(device),
            audit_batch["byte_attention_mask"].to(device),
            audit_batch["questions"],
            audit_batch["answers"],
            answer_only_loss=True,
        )
    label_audit = audit_answer_only_labels(
        audit_output.prepared, audit_batch["answers"], model.llm_tokenizer
    )

    git = git_metadata()
    manifest_hashes = {
        "train": sha256_file(args.train_manifest),
        "valid": sha256_file(args.valid_manifest),
    }
    config = {
        "task": "SpatialSense answer-only yes/no causal LM pilot",
        "loss": "full-vocabulary causal LM cross entropy over one yes/no token",
        "eos_supervised": False,
        "epochs": 1,
        "micro_batch_size": 1,
        "gradient_accumulation": 8,
        "effective_batch_size": 8,
        "optimizer_steps": optimizer_steps,
        "precision": "FP16 autocast with GradScaler",
        "seed": args.seed,
        "pretrained_lr": args.pretrained_lr,
        "new_lr": args.new_lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "gradient_clipping": args.gradient_clipping,
        "byteformer_frozen": True,
        "vicuna_frozen": True,
        "fresh_blip2_stage1_initialization": True,
        "prior_bridge_checkpoint": None,
        "train_manifest": str(args.train_manifest.resolve()),
        "valid_manifest": str(args.valid_manifest.resolve()),
        "cache_manifest_sha256": manifest_hashes,
        "model_load": load_report,
        "optimizer_groups": optimizer_report,
        "label_audit": label_audit,
        "git_at_launch": git,
        "test_split_used": False,
    }
    write_json(args.output_dir / "training_config.json", config)
    checkpoint_metadata = {
        "training_config": config,
        "random_seed": args.seed,
        "cache_manifest_sha256": manifest_hashes,
    }

    validation_steps = {0, math.ceil(optimizer_steps / 2), optimizer_steps}
    validations: dict[str, Any] = {}
    best_accuracy = best_auroc = None
    best_step = None

    def run_validation(step: int) -> None:
        nonlocal best_accuracy, best_auroc, best_step
        evaluation = evaluate_answer_only(
            model,
            valid_dataset,
            device=device,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
        )
        path = args.output_dir / f"validation_step_{step}.json"
        write_json(path, evaluation)
        validations[str(step)] = {"path": str(path.resolve()), **summary(evaluation)}
        metric = evaluation["logit_yes_minus_no"]
        if checkpoint_is_better(
            metric["accuracy"], metric["auroc"], best_accuracy, best_auroc
        ):
            best_accuracy, best_auroc, best_step = (
                metric["accuracy"], metric["auroc"], step
            )
            atomic_torch_save(
                bridge_artifact(
                    model,
                    step,
                    {
                        **checkpoint_metadata,
                        "selection_metric": {
                            "accuracy": best_accuracy,
                            "auroc": best_auroc,
                        },
                    },
                ),
                args.output_dir / "bridge_best.pt",
            )

    pilot_started = time.perf_counter()
    run_validation(0)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        collate_fn=collate_cached_features,
        pin_memory=args.num_workers > 0,
    )
    curve_path = args.output_dir / "training_curve.jsonl"
    curve_path.write_text("", encoding="utf-8")
    optimizer.zero_grad(set_to_none=True)
    model.train()
    micro_step = optimizer_step = nonfinite_events = 0
    accumulated_loss = 0.0
    accumulated_samples = 0
    interval_started = epoch_started = time.perf_counter()
    training_compute_seconds = 0.0

    for batch in train_loader:
        group_start = (micro_step // args.gradient_accumulation) * args.gradient_accumulation
        group_size = min(args.gradient_accumulation, len(train_dataset) - group_start)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model.forward_from_byte_features(
                batch["byte_features"].to(device, non_blocking=True),
                batch["byte_attention_mask"].to(device, non_blocking=True),
                batch["questions"],
                batch["answers"],
                answer_only_loss=True,
            )
            if not bool(torch.isfinite(output.loss).item()):
                nonfinite_events += 1
                raise RuntimeError(f"Non-finite loss at micro step {micro_step}")
            scaled_loss = output.loss / group_size
        scaler.scale(scaled_loss).backward()
        accumulated_loss += float(output.loss.detach().float().item())
        accumulated_samples += 1
        micro_step += 1
        if micro_step % args.gradient_accumulation and micro_step != len(train_dataset):
            continue

        scaler.unscale_(optimizer)
        gradients = gradient_report(model.bridge)
        total_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.bridge.parameters(), args.gradient_clipping).item()
        )
        if not math.isfinite(total_gradient_norm) or not all(
            math.isfinite(value) for value in gradients.values()
        ):
            nonfinite_events += 1
            raise RuntimeError(f"Non-finite gradient at step {optimizer_step + 1}")
        learning_rates = {
            group["group_name"]: group["lr"] for group in optimizer.param_groups
        }
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
        now = time.perf_counter()
        training_compute_seconds += now - interval_started
        row = {
            "optimizer_step": optimizer_step,
            "micro_step": micro_step,
            "mean_loss": accumulated_loss / accumulated_samples,
            "learning_rates_used": learning_rates,
            "gradient_norms": gradients,
            "total_gradient_norm_before_clipping": total_gradient_norm,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "elapsed_seconds": now - pilot_started,
            "nonfinite_events": nonfinite_events,
        }
        with curve_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        accumulated_loss = 0.0
        accumulated_samples = 0
        interval_started = now
        if optimizer_step in validation_steps:
            run_validation(optimizer_step)
            model.train()
            interval_started = time.perf_counter()

    if (micro_step, optimizer_step) != (len(train_dataset), optimizer_steps):
        raise RuntimeError("One-epoch step accounting failed")
    epoch_seconds = time.perf_counter() - epoch_started
    atomic_torch_save(
        bridge_artifact(model, optimizer_step, {**checkpoint_metadata, "kind": "last"}),
        args.output_dir / "bridge_last.pt",
    )
    atomic_torch_save(
        {
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "optimizer_step": optimizer_step,
            "metadata": checkpoint_metadata,
        },
        args.output_dir / "optimizer_scheduler_last.pt",
    )

    train_evaluation = evaluate_answer_only(
        model,
        train_dataset,
        device=device,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    write_json(args.output_dir / "train_epoch_end.json", train_evaluation)

    visual_keys = [row["visual_cache_key"] for row in valid_dataset.records]
    permutation = fixed_derangement(len(valid_dataset), args.seed, visual_keys)
    write_json(
        args.output_dir / "valid_shuffled_byte_mapping.json",
        {"seed": args.seed, "mapping": permutation},
    )
    zero_evaluation = evaluate_answer_only(
        model,
        valid_dataset,
        device=device,
        batch_size=args.eval_batch_size,
        projected_query_scale=0.0,
        num_workers=args.num_workers,
    )
    shuffled_evaluation = evaluate_answer_only(
        model,
        PermutedVisualDataset(valid_dataset, permutation),
        device=device,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    write_json(args.output_dir / "validation_zero_byte.json", zero_evaluation)
    write_json(args.output_dir / "validation_shuffled_byte.json", shuffled_evaluation)

    frozen = {}
    for name, module in (("byteformer", model.byteformer_adapter), ("vicuna", model.vicuna)):
        frozen[name] = {
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in module.parameters() if parameter.requires_grad
            ),
            "parameters_with_grad": sum(
                parameter.grad is not None for parameter in module.parameters()
            ),
        }
    if any(row["trainable_parameter_count"] or row["parameters_with_grad"] for row in frozen.values()):
        raise RuntimeError(f"Frozen module audit failed: {frozen}")

    final_valid = validations[str(optimizer_steps)]
    ablation = {
        "normal": {
            "single_token_generation": final_valid["single_token_generation"],
            "logit_yes_minus_no": final_valid["logit_yes_minus_no"],
        },
        "zero_byte": summary(zero_evaluation),
        "shuffled_byte": summary(shuffled_evaluation),
    }
    write_json(args.output_dir / "epoch_end_ablation_comparison.json", ablation)
    report = {
        "task": config["task"],
        "config": config,
        "label_audit": label_audit,
        "memory_after_model_load_bytes": memory_after_model_load,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "epoch_duration_seconds": epoch_seconds,
        "training_compute_seconds_excluding_validation": training_compute_seconds,
        "total_pilot_duration_seconds": time.perf_counter() - pilot_started,
        "optimizer_steps": optimizer_step,
        "micro_steps": micro_step,
        "nonfinite_events": nonfinite_events,
        "validations": validations,
        "train_epoch_end": summary(train_evaluation),
        "best_checkpoint_step": best_step,
        "best_accuracy": best_accuracy,
        "best_auroc": best_auroc,
        "epoch_end_ablation": ablation,
        "frozen_modules": frozen,
        "test_split_used": False,
        "artifacts": {
            "best_bridge": str((args.output_dir / "bridge_best.pt").resolve()),
            "last_bridge": str((args.output_dir / "bridge_last.pt").resolve()),
            "training_curve": str(curve_path.resolve()),
        },
    }
    write_json(args.output_dir / "training_report.json", report)
    print(json.dumps({"status": "complete", "report": str((args.output_dir / "training_report.json").resolve())}))


if __name__ == "__main__":
    main()
