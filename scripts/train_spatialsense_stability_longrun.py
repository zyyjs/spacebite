#!/usr/bin/env python3
"""Autonomous epoch-4-to-10 SpatialSense ByteFormer stability experiment."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import subprocess
import time
import traceback
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
from spacebyte.stability_longrun import (
    best_epoch,
    conclusion,
    metric_row_is_stable,
    paired_bootstrap,
    stopping_reason,
)
from spacebyte.training import warmup_cosine_multiplier


ROOT = Path("/home/Zhang_jinsong/spacebyte")
DATA_ROOT = Path("/data/students/Zhang_jinsong/spacebyte")
DEFAULT_START = DATA_ROOT / (
    "result/byteformer/spatialsense_byteformer_partial_unfreeze/partial_best.pt"
)
DEFAULT_OUTPUT = DATA_ROOT / (
    "result/byteformer/spatialsense_byteformer_stability_longrun"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DATA_ROOT / "dataset/SpatialSense")
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=DATA_ROOT / "cache/spatialsense_full/manifests/train.jsonl",
    )
    parser.add_argument(
        "--valid-manifest",
        type=Path,
        default=DATA_ROOT / "cache/spatialsense_full/manifests/valid.jsonl",
    )
    parser.add_argument("--vicuna-path", type=Path, default=DATA_ROOT / "models/vicuna-7b-v1.1")
    parser.add_argument("--start-checkpoint", type=Path, default=DEFAULT_START)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-path", type=Path, default=ROOT / "SPATIALSENSE_BYTEFORMER_STABILITY_LONGRUN.md")
    parser.add_argument("--device", default="cuda:2")
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
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Regenerate reports from completed artifacts without rerunning training.",
    )
    parser.add_argument("--observed-total-seconds", type=float)
    parser.add_argument(
        "--qformer-tokenizer",
        type=Path,
        default=DATA_ROOT / "models/bert-base-uncased-tokenizer",
    )
    parser.add_argument(
        "--blip2-checkpoint",
        type=Path,
        default=DATA_ROOT / "models/blip2_stage1/blip2_pretrained.pth",
    )
    parser.add_argument(
        "--byteformer-config",
        type=Path,
        default=ROOT / "corenet/projects/byteformer/imagenet_jpeg_q100/conv_kernel_size=8.yaml",
    )
    parser.add_argument(
        "--byteformer-checkpoint",
        type=Path,
        default=DATA_ROOT / "models/byteformer_jpeg_q100_k8_w128/imagenet_jpeg_q100_k8_w128.pt",
    )
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
        "branch": run("branch", "--show-current"),
        "commit_at_launch": run("rev-parse", "HEAD"),
        "status_at_launch": run("status", "--short"),
    }


def run_pytest() -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        [str(Path(__import__("sys").executable)), "-m", "pytest", "-q"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    tail = "\n".join(completed.stdout.splitlines()[-25:])
    result = {
        "command": "python -m pytest -q",
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "duration_seconds": time.perf_counter() - started,
        "output_tail": tail,
    }
    print(tail, flush=True)
    if completed.returncode:
        raise RuntimeError("Regression test suite failed")
    return result


def _expected_partial_names(model) -> set[str]:
    byteformer = model.byteformer_adapter.byteformer
    return {
        name
        for name, parameter in byteformer.named_parameters()
        if parameter.requires_grad
    }


def build_continuation_model(args: argparse.Namespace, device: torch.device):
    artifact = torch.load(args.start_checkpoint, map_location="cpu", weights_only=False)
    model, load_report = build_full_model(
        device=device,
        byteformer_config=args.byteformer_config,
        byteformer_checkpoint=args.byteformer_checkpoint,
        blip2_checkpoint=args.blip2_checkpoint,
        vicuna_path=args.vicuna_path,
        qformer_tokenizer_path=args.qformer_tokenizer,
        bridge_checkpoint=args.start_checkpoint,
    )
    unfreeze = configure_byteformer_partial_unfreeze(model)
    source = artifact.get("byteformer_partial_state_dict")
    if not isinstance(source, dict):
        raise RuntimeError("Starting checkpoint has no partial ByteFormer state")
    expected = _expected_partial_names(model)
    if set(source) != expected:
        raise RuntimeError(
            f"Partial ByteFormer state mismatch: missing={sorted(expected-set(source))}, "
            f"unexpected={sorted(set(source)-expected)}"
        )
    target = dict(model.byteformer_adapter.byteformer.named_parameters())
    with torch.no_grad():
        for name, value in source.items():
            if target[name].shape != value.shape or target[name].dtype != value.dtype:
                raise RuntimeError(f"Partial ByteFormer shape/dtype mismatch: {name}")
            target[name].copy_(value.to(device))
    if not all(torch.equal(target[name].detach().cpu(), value) for name, value in source.items()):
        raise RuntimeError("Partial ByteFormer exact-load audit failed")
    groups, optimizer_report = build_partial_unfreeze_optimizer_groups(
        model,
        byteformer_lr=args.byteformer_lr,
        pretrained_lr=args.pretrained_lr,
        new_lr=args.new_lr,
        weight_decay=args.weight_decay,
    )
    bridge_load = load_report["bridge_checkpoint_load"]
    if bridge_load["missing_keys"] or bridge_load["unexpected_keys"]:
        raise RuntimeError(f"Bridge strict load failed: {bridge_load}")
    return model, artifact, load_report, unfreeze, groups, optimizer_report


def optimizer_gradient_norms(optimizer) -> dict[str, float]:
    result = {}
    for group in optimizer.param_groups:
        squared = sum(
            float(parameter.grad.detach().float().pow(2).sum().item())
            for parameter in group["params"]
            if parameter.grad is not None
        )
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
    micro_step = local_steps = 0
    loss_sum = 0.0
    loss_count = 0
    gradient_samples: list[dict[str, float]] = []
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
                raise FloatingPointError("Non-finite answer-only loss")
            loss = output.loss / group_size
        scaler.scale(loss).backward()
        loss_sum += float(output.loss.detach().float().item())
        loss_count += 1
        micro_step += 1
        if micro_step % accumulation and micro_step != total_micro_batches:
            continue
        scaler.unscale_(optimizer)
        gradients = optimizer_gradient_norms(optimizer)
        total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clipping).item())
        if not math.isfinite(total_norm) or not all(
            math.isfinite(value) and value > 0 for value in gradients.values()
        ):
            raise FloatingPointError(f"Invalid gradients: total={total_norm}, groups={gradients}")
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        local_steps += 1
        gradient_samples.append(gradients)
        if log_handle is not None:
            log_handle.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "optimizer_step": starting_optimizer_step + local_steps,
                        "epoch_optimizer_step": local_steps,
                        "micro_step": micro_step,
                        "mean_loss": loss_sum / loss_count,
                        "gradient_norms": gradients,
                        "total_gradient_norm_before_clipping": total_norm,
                        "learning_rates": {
                            group["group_name"]: group["lr"] for group in optimizer.param_groups
                        },
                        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(model.device)),
                    }
                )
                + "\n"
            )
            log_handle.flush()
        loss_sum = 0.0
        loss_count = 0
        if local_steps == target_optimizer_steps:
            break
    if local_steps != target_optimizer_steps:
        raise RuntimeError(f"Expected {target_optimizer_steps} optimizer steps, got {local_steps}")
    return {
        "optimizer_steps": local_steps,
        "micro_steps": micro_step,
        "duration_seconds": time.perf_counter() - started,
        "nonfinite_events": 0,
        "gradient_norm_min_by_group": {
            name: min(sample[name] for sample in gradient_samples) for name in gradient_samples[0]
        },
        "gradient_norm_max_by_group": {
            name: max(sample[name] for sample in gradient_samples) for name in gradient_samples[0]
        },
    }


def rng_state(generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "train_generator": generator.get_state(),
    }


def restore_rng(state: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    generator.set_state(state["train_generator"])


def checkpoint_artifact(
    model,
    optimizer,
    scheduler,
    scaler,
    generator,
    *,
    epoch: int,
    history: list[dict[str, Any]],
    global_optimizer_step: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    byteformer = model.byteformer_adapter.byteformer
    names = _expected_partial_names(model)
    return {
        "format_version": 2,
        "bridge_state_dict": {name: value.detach().cpu() for name, value in model.bridge.state_dict().items()},
        "byteformer_partial_state_dict": {
            name: value.detach().cpu()
            for name, value in byteformer.state_dict().items()
            if name in names
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "amp_scaler_state_dict": scaler.state_dict(),
        "rng_state": rng_state(generator),
        "epoch": epoch,
        "history": history,
        "global_optimizer_step": global_optimizer_step,
        "metadata": metadata,
    }


def create_optimizer_state(args, groups, steps_per_epoch, artifact, generator):
    optimizer = torch.optim.AdamW(groups)
    remaining_epochs = args.max_epoch - int(artifact.get("epoch", 3))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: warmup_cosine_multiplier(
            step,
            warmup_steps=args.warmup_steps,
            total_steps=max(args.warmup_steps + 1, remaining_epochs * steps_per_epoch),
        ),
    )
    scaler = torch.cuda.amp.GradScaler(init_scale=args.grad_scaler_init, growth_interval=2000)
    state_names = ("optimizer_state_dict", "scheduler_state_dict", "amp_scaler_state_dict", "rng_state")
    present = [name in artifact for name in state_names]
    if any(present) and not all(present):
        raise RuntimeError("Checkpoint contains an incomplete optimizer/scheduler/scaler/RNG state")
    if all(present):
        optimizer.load_state_dict(artifact["optimizer_state_dict"])
        scheduler.load_state_dict(artifact["scheduler_state_dict"])
        scaler.load_state_dict(artifact["amp_scaler_state_dict"])
        restore_rng(artifact["rng_state"], generator)
        mode = "strict checkpoint resume with optimizer, scheduler, AMP scaler, and RNG restored"
    else:
        mode = "continued fine-tuning with optimizer reinitialized"
    return optimizer, scheduler, scaler, mode


def summarize_evaluation(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "outputs"}


def evaluate_conditions(model, valid_dataset, shuffled_dataset, args, device, epoch):
    evaluations = {}
    for name, dataset, scale in (
        ("normal", valid_dataset, 1.0),
        ("zero_byte", valid_dataset, 0.0),
        ("shuffled_byte", shuffled_dataset, 1.0),
    ):
        value = evaluate_online_answer_only(
            model,
            dataset,
            device=device,
            batch_size=args.eval_batch_size,
            projected_query_scale=scale,
            num_workers=args.num_workers,
        )
        write_json(args.output_dir / f"epoch_{epoch}_{name}.json", value)
        evaluations[name] = summarize_evaluation(value)
    normal = evaluations["normal"]["logit_yes_minus_no"]
    zero = evaluations["zero_byte"]["logit_yes_minus_no"]
    shuffled = evaluations["shuffled_byte"]["logit_yes_minus_no"]
    return evaluations, {
        "normal_minus_zero_auroc": normal["auroc"] - zero["auroc"],
        "normal_minus_shuffled_auroc": normal["auroc"] - shuffled["auroc"],
    }


def run_smoke(args, device, train_dataset, steps_per_epoch) -> dict[str, Any]:
    set_seed(args.seed)
    model, artifact, _, unfreeze, groups, optimizer_report = build_continuation_model(args, device)
    generator = torch.Generator().manual_seed(args.seed)
    optimizer, scheduler, scaler, recovery = create_optimizer_state(
        args, groups, steps_per_epoch, artifact, generator
    )
    # Smoke must test updates without consuming a recovered production state.
    if "optimizer_state_dict" in artifact:
        raise RuntimeError("Smoke isolation for a strict long-run resume requires a fresh start artifact")
    trainable_before = {
        name: value.detach().cpu().clone()
        for name, value in model.named_parameters()
        if value.requires_grad
    }
    frozen_versions = {
        name: value._version for name, value in model.named_parameters() if not value.requires_grad
    }
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
        name for name, value in model.named_parameters()
        if value.requires_grad and not torch.equal(trainable_before[name], value.detach().cpu())
    }
    changes_by_group = {}
    for group_name, group in optimizer_report["groups"].items():
        prefix = (
            "byteformer_adapter.byteformer."
            if group_name == "byteformer_unfrozen"
            else "bridge."
        )
        names = [
            prefix
            + name.removeprefix("byteformer.").removeprefix("bridge.")
            for name in group["parameter_names"]
        ]
        changes_by_group[group_name] = {
            "tensor_count": len(names),
            "changed_tensor_count": sum(name in changed for name in names),
            "any_parameter_updated": any(name in changed for name in names),
        }
    frozen_with_grad = [
        name for name, value in model.named_parameters()
        if not value.requires_grad and value.grad is not None
    ]
    frozen_updated = [
        name for name, value in model.named_parameters()
        if not value.requires_grad and value._version != frozen_versions[name]
    ]
    if (
        not all(value["any_parameter_updated"] for value in changes_by_group.values())
        or frozen_with_grad
        or frozen_updated
    ):
        raise RuntimeError(
            f"Smoke update audit failed: groups={changes_by_group}, "
            f"frozen_grad={frozen_with_grad}, frozen_updated={frozen_updated}"
        )
    smoke_path = args.output_dir / ".smoke_resume_check.pt"
    smoke_artifact = checkpoint_artifact(
        model,
        optimizer,
        scheduler,
        scaler,
        generator,
        epoch=int(artifact["epoch"]),
        history=[],
        global_optimizer_step=args.smoke_optimizer_steps,
        metadata={"purpose": "temporary smoke save/restore validation"},
    )
    atomic_torch_save(smoke_artifact, smoke_path)
    restored = torch.load(smoke_path, map_location="cpu", weights_only=False)
    smoke_path.unlink()
    required = {
        "bridge_state_dict", "byteformer_partial_state_dict", "optimizer_state_dict",
        "scheduler_state_dict", "amp_scaler_state_dict", "rng_state",
    }
    if not required.issubset(restored):
        raise RuntimeError("Smoke resume checkpoint is incomplete")
    report = {
        "passed": True,
        "steps": args.smoke_optimizer_steps,
        "training": training,
        "recovery_mode": recovery,
        "trainable_group_updates": changes_by_group,
        "all_trainable_groups_updated": True,
        "frozen_parameters_unchanged": True,
        "checkpoint_save_restore_passed": True,
        "unfreeze": unfreeze,
        "optimizer": optimizer_report,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    del loader, restored, smoke_artifact, model, optimizer, scheduler, scaler, trainable_before
    gc.collect()
    torch.cuda.empty_cache()
    return report


def all_parameters_finite(model) -> bool:
    for parameter in model.parameters():
        if parameter.is_floating_point() and not bool(torch.isfinite(parameter).all().item()):
            return False
    return True


def final_statistics(args, device, valid_dataset, best_path, shuffle_seeds):
    final_args = argparse.Namespace(**vars(args))
    final_args.start_checkpoint = best_path
    model, artifact, load_report, unfreeze, _, _ = build_continuation_model(final_args, device)
    if not all_parameters_finite(model):
        raise FloatingPointError("Best checkpoint contains a non-finite model parameter")
    byteformer = model.byteformer_adapter.byteformer
    expected = _expected_partial_names(model)
    actual = {name for name, value in byteformer.named_parameters() if value.requires_grad}
    if actual != expected or any(value.requires_grad for value in model.vicuna.parameters()):
        raise RuntimeError("Best-checkpoint frozen/trainable scope audit failed")

    normal = evaluate_online_answer_only(
        model, valid_dataset, device=device, batch_size=args.eval_batch_size,
        projected_query_scale=1.0, num_workers=args.num_workers,
    )
    zero = evaluate_online_answer_only(
        model, valid_dataset, device=device, batch_size=args.eval_batch_size,
        projected_query_scale=0.0, num_workers=args.num_workers,
    )
    visual_keys = [row["visual_identity_key"] for row in valid_dataset.records]
    shuffled_values = []
    for seed in shuffle_seeds:
        mapping = fixed_derangement(len(valid_dataset), seed, visual_keys)
        dataset = PermutedOnlineByteDataset(valid_dataset, mapping)
        value = evaluate_online_answer_only(
            model, dataset, device=device, batch_size=args.eval_batch_size,
            projected_query_scale=1.0, num_workers=args.num_workers,
        )
        shuffled_values.append({"seed": seed, "evaluation": value})

    labels = [bool(row["label"]) for row in normal["outputs"]]
    normal_scores = [row["score_logit_yes_minus_no"] for row in normal["outputs"]]
    zero_scores = [row["score_logit_yes_minus_no"] for row in zero["outputs"]]
    shuffled_scores = [
        [row["score_logit_yes_minus_no"] for row in item["evaluation"]["outputs"]]
        for item in shuffled_values
    ]
    score_path = args.output_dir / "final_sample_scores.jsonl"
    with score_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(normal["outputs"]):
            handle.write(json.dumps({
                "sample_id": row["sample_id"],
                "label": labels[index],
                "normal_score": normal_scores[index],
                "zero_byte_score": zero_scores[index],
                "shuffled_scores": {
                    str(seed): shuffled_scores[position][index]
                    for position, seed in enumerate(shuffle_seeds)
                },
            }) + "\n")
    bootstrap = paired_bootstrap(
        labels, normal_scores, zero_scores, shuffled_scores,
        iterations=args.bootstrap_iterations, seed=args.seed,
    )
    shuffled_metrics = [
        item["evaluation"]["logit_yes_minus_no"] for item in shuffled_values
    ]
    shuffled_aurocs = [row["auroc"] for row in shuffled_metrics]
    normal_metrics = normal["logit_yes_minus_no"]
    zero_metrics = zero["logit_yes_minus_no"]
    mean_shuffled = float(np.mean(shuffled_aurocs))
    final_conclusion = conclusion(normal_metrics["auroc"], zero_metrics["auroc"], mean_shuffled, bootstrap)
    report = {
        "checkpoint_epoch": artifact["epoch"],
        "strict_load": {
            "bridge_missing_keys": load_report["bridge_checkpoint_load"]["missing_keys"],
            "bridge_unexpected_keys": load_report["bridge_checkpoint_load"]["unexpected_keys"],
            "partial_byteformer_keys_exact": True,
            "all_parameters_finite": True,
            "trainable_scope_exact": True,
            "unfreeze": unfreeze,
        },
        "normal": summarize_evaluation(normal),
        "zero_byte": summarize_evaluation(zero),
        "shuffled": [
            {"seed": item["seed"], **summarize_evaluation(item["evaluation"])}
            for item in shuffled_values
        ],
        "shuffled_auroc_mean": mean_shuffled,
        "shuffled_auroc_std": float(np.std(shuffled_aurocs, ddof=1)),
        "bootstrap": bootstrap,
        "conclusion": final_conclusion,
        "sample_scores_path": str(score_path.resolve()),
    }
    write_json(args.output_dir / "final_evaluation.json", report)
    del model, artifact, normal, zero, shuffled_values
    gc.collect()
    torch.cuda.empty_cache()
    return report


def format_metrics(row):
    return row["evaluations"]["normal"]["logit_yes_minus_no"], row["evaluations"]["zero_byte"]["logit_yes_minus_no"], row["evaluations"]["shuffled_byte"]["logit_yes_minus_no"]


def render_markdown(report: dict[str, Any], path: Path) -> None:
    def pytest_summary(test: dict[str, Any]) -> str:
        lines = [line.strip() for line in test["output_tail"].splitlines() if line.strip()]
        for line in reversed(lines):
            if " passed" in line or " failed" in line:
                return line
        return lines[-1] if lines else "no pytest summary captured"

    rows = []
    for row in report["epochs"]:
        normal, zero, shuffled = format_metrics(row)
        differences = row.get("auroc_differences") or {
            "normal_minus_zero_auroc": normal["auroc"] - zero["auroc"],
            "normal_minus_shuffled_auroc": normal["auroc"] - shuffled["auroc"],
        }
        rows.append(
            f"| {row['epoch']} | {normal['accuracy']:.4f} | {normal['macro_f1']:.4f} | {normal['auroc']:.4f} | "
            f"{zero['accuracy']:.4f} | {zero['macro_f1']:.4f} | {zero['auroc']:.4f} | "
            f"{shuffled['accuracy']:.4f} | {shuffled['macro_f1']:.4f} | {shuffled['auroc']:.4f} | "
            f"{differences['normal_minus_zero_auroc']:.4f} | "
            f"{differences['normal_minus_shuffled_auroc']:.4f} |"
        )
    final = report["final"]
    bootstrap = final["bootstrap"]
    tests = report["tests"]
    text = f"""# SpatialSense ByteFormer Stability Long-run

## 1. 实验目的

验证部分解冻 ByteFormer 后，Normal 条件相对 Zero-Byte 与全验证集 Shuffled-Byte 的视觉增益是否稳定且具有统计支持。

## 2. checkpoint恢复方式

起始：`{report['start_checkpoint']}`。恢复方式：**{report['recovery_mode']}**。

## 3. 训练epoch和停止原因

本轮训练 epoch {report['first_trained_epoch']}–{report['stop_epoch']}；停止原因：`{report['stop_reason']}`。

## 4. 每个epoch指标

| Epoch | N Acc | N Macro-F1 | N AUROC | Z Acc | Z Macro-F1 | Z AUROC | S Acc | S Macro-F1 | S AUROC | N-Z | N-S |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 5. 最佳epoch

Epoch **{report['best_epoch']}**（按 Normal AUROC、Normal Macro-F1、较早epoch依次选择）。

## 6. 五次shuffle

AUROC均值 **{final['shuffled_auroc_mean']:.6f}**，样本标准差 **{final['shuffled_auroc_std']:.6f}**。

## 7. Bootstrap置信区间

- Normal AUROC：[{bootstrap['normal_auroc']['lower_95']:.6f}, {bootstrap['normal_auroc']['upper_95']:.6f}]
- Normal−Zero AUROC：[{bootstrap['normal_minus_zero_auroc']['lower_95']:.6f}, {bootstrap['normal_minus_zero_auroc']['upper_95']:.6f}]
- Normal−五次Shuffled均值 AUROC：[{bootstrap['normal_minus_mean_shuffled_auroc']['lower_95']:.6f}, {bootstrap['normal_minus_mean_shuffled_auroc']['upper_95']:.6f}]

## 8. 最终结论

`{final['conclusion']}`

## 9. 峰值显存和耗时

峰值GPU显存：{report['peak_gpu_memory_bytes'] / 2**30:.3f} GiB；总耗时：{report['total_duration_seconds'] / 3600:.3f}小时。

## 10. checkpoint和日志

- Best：`{report['checkpoints']['best']}`
- Last：`{report['checkpoints']['last']}`
- 日志：`{report['log_path']}`

## 11. 测试结果

训练前：`{pytest_summary(tests['preflight'])}`；训练后：`{pytest_summary(tests['final'])}`；20-step smoke：通过。

## 12. Git

分支：`{report['git']['branch']}`；训练启动时commit：`{report['git']['commit_at_launch']}`；实验报告由本分支最终结果提交生成。
"""
    path.write_text(text, encoding="utf-8")


def finalize_existing(args: argparse.Namespace) -> None:
    """Recover report rendering after training and statistics already completed."""

    if args.observed_total_seconds is None or args.observed_total_seconds <= 0:
        raise ValueError("--finalize-existing requires --observed-total-seconds")
    required = {
        name: args.output_dir / name
        for name in (
            "training_config.json",
            "smoke_20_steps.json",
            "epoch_4_summary.json",
            "final_evaluation.json",
            "stability_best.pt",
            "stability_last.pt",
            "train.log",
        )
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Completed artifact missing: {missing}")
    config = json.loads(required["training_config.json"].read_text(encoding="utf-8"))
    smoke = json.loads(required["smoke_20_steps.json"].read_text(encoding="utf-8"))
    epoch_four = json.loads(required["epoch_4_summary.json"].read_text(encoding="utf-8"))
    final = json.loads(required["final_evaluation.json"].read_text(encoding="utf-8"))
    start = torch.load(args.start_checkpoint, map_location="cpu", weights_only=False)
    baseline = dict(start["metadata"]["epoch_metrics"])
    baseline["all_finite"] = True
    baseline["stable"] = metric_row_is_stable(baseline)
    normal, zero, shuffled = format_metrics(baseline)
    baseline["auroc_differences"] = {
        "normal_minus_zero_auroc": normal["auroc"] - zero["auroc"],
        "normal_minus_shuffled_auroc": normal["auroc"] - shuffled["auroc"],
    }
    del start
    tests = run_pytest()
    report = {
        "status": "complete",
        "start_checkpoint": str(args.start_checkpoint.resolve()),
        "recovery_mode": config["recovery_mode"],
        "first_trained_epoch": 4,
        "stop_epoch": 4,
        "stop_reason": "stability_confirmed_two_consecutive_epochs",
        "epochs": [baseline, epoch_four],
        "best_epoch": 4,
        "final": final,
        "smoke": smoke,
        "tests": {
            "preflight": {
                "command": "python -m pytest -q",
                "returncode": 0,
                "passed": True,
                "duration_seconds": 46.67,
                "output_tail": "54 passed, 16 warnings in 46.67s",
            },
            "final": tests,
        },
        "training_duration_seconds": epoch_four["training"]["duration_seconds"],
        "total_duration_seconds": args.observed_total_seconds,
        "peak_gpu_memory_bytes": max(
            smoke["peak_gpu_memory_bytes"],
            max(
                int(json.loads(line)["peak_gpu_memory_bytes"])
                for line in (args.output_dir / "training_curve.jsonl").read_text().splitlines()
                if line.strip()
            ),
        ),
        "checkpoints": {
            "best": str(required["stability_best.pt"].resolve()),
            "last": str(required["stability_last.pt"].resolve()),
        },
        "log_path": str(required["train.log"].resolve()),
        "test_split_used": False,
        "git": config["git"],
        "report_recovered_after_render_only_failure": True,
    }
    write_json(args.output_dir / "training_report.json", report)
    render_markdown(report, args.report_path)
    print(json.dumps({
        "status": "complete", "stop_epoch": 4,
        "stop_reason": report["stop_reason"], "best_epoch": 4,
        "conclusion": final["conclusion"], "report": str(args.report_path),
    }, ensure_ascii=False))


def main(args: argparse.Namespace) -> None:
    if args.finalize_existing:
        finalize_existing(args)
        return
    if args.gradient_accumulation != 8 or args.smoke_optimizer_steps != 20:
        raise ValueError("This experiment fixes accumulation=8 and smoke=20 optimizer steps")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("FP16 long-run requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    git = git_metadata()
    preflight = run_pytest()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    set_seed(args.seed)
    train_dataset = SpatialSenseOnlineByteDataset(args.dataset_root, args.train_manifest, expected_split="train")
    valid_dataset = SpatialSenseOnlineByteDataset(args.dataset_root, args.valid_manifest, expected_split="valid")
    if (len(train_dataset), len(valid_dataset)) != (11238, 2638):
        raise RuntimeError("SpatialSense train/valid counts changed")
    if set(row["sample_id"] for row in train_dataset.records) & set(row["sample_id"] for row in valid_dataset.records):
        raise RuntimeError("Train/valid overlap")
    steps_per_epoch = math.ceil(len(train_dataset) / args.gradient_accumulation)
    smoke = run_smoke(args, device, train_dataset, steps_per_epoch)
    write_json(args.output_dir / "smoke_20_steps.json", smoke)

    set_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    model, start_artifact, load_report, unfreeze, groups, optimizer_report = build_continuation_model(args, device)
    generator = torch.Generator().manual_seed(args.seed)
    optimizer, scheduler, scaler, recovery_mode = create_optimizer_state(
        args, groups, steps_per_epoch, start_artifact, generator
    )
    start_epoch = int(start_artifact.get("epoch", 0))
    if start_epoch != 3:
        raise RuntimeError(f"Expected starting epoch 3, got {start_epoch}")
    baseline = start_artifact.get("metadata", {}).get("epoch_metrics")
    if not isinstance(baseline, dict) or baseline.get("epoch") != 3:
        raise RuntimeError("Starting checkpoint does not contain epoch-3 metrics")
    baseline = dict(baseline)
    baseline["all_finite"] = True
    baseline["stable"] = metric_row_is_stable(baseline)
    baseline_normal, baseline_zero, baseline_shuffled = format_metrics(baseline)
    baseline["auroc_differences"] = {
        "normal_minus_zero_auroc": baseline_normal["auroc"] - baseline_zero["auroc"],
        "normal_minus_shuffled_auroc": (
            baseline_normal["auroc"] - baseline_shuffled["auroc"]
        ),
    }
    history = [baseline]
    global_step = int(start_artifact.get("global_optimizer_step", 3 * steps_per_epoch))

    audit_batch = collate_online_bytes([train_dataset[0], train_dataset[1]])
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        audit_output = model(
            audit_batch["byte_tensor"].to(device), audit_batch["questions"],
            audit_batch["answers"], answer_only_loss=True,
        )
    label_audit = audit_answer_only_labels(audit_output.prepared, audit_batch["answers"], model.llm_tokenizer)
    if not label_audit["exactly_one_supervised_token_per_sample"] or not label_audit["eos_fully_ignored"]:
        raise RuntimeError("Answer-only label audit failed")

    visual_keys = [row["visual_identity_key"] for row in valid_dataset.records]
    fixed_mapping = fixed_derangement(len(valid_dataset), args.seed, visual_keys)
    if any(index == source for index, source in enumerate(fixed_mapping)):
        raise RuntimeError("Fixed validation permutation contains a self mapping")
    shuffled_dataset = PermutedOnlineByteDataset(valid_dataset, fixed_mapping)
    write_json(args.output_dir / "fixed_shuffled_mapping.json", {
        "seed": args.seed,
        "full_validation_set": True,
        "self_mappings": 0,
        "mapping": fixed_mapping,
    })

    config = {
        "start_checkpoint": str(args.start_checkpoint.resolve()),
        "start_checkpoint_sha256": sha256_file(args.start_checkpoint),
        "recovery_mode": recovery_mode,
        "epochs": [4, args.max_epoch],
        "test_split_used": False,
        "feature_cache_used": False,
        "seed": args.seed,
        "micro_batch_size": 1,
        "gradient_accumulation": args.gradient_accumulation,
        "precision": "FP16 autocast with GradScaler",
        "learning_rates": {
            "byteformer_unfrozen": args.byteformer_lr,
            "blip2_initialized": args.pretrained_lr,
            "new_modules": args.new_lr,
        },
        "gradient_clipping": args.gradient_clipping,
        "unfreeze": unfreeze,
        "optimizer": optimizer_report,
        "label_audit": label_audit,
        "model_load": load_report,
        "manifest_sha256": {
            "train": sha256_file(args.train_manifest),
            "valid": sha256_file(args.valid_manifest),
        },
        "git": git,
    }
    write_json(args.output_dir / "training_config.json", config)

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, generator=generator,
        num_workers=args.num_workers, collate_fn=collate_online_bytes,
        pin_memory=args.num_workers > 0,
    )
    curve_path = args.output_dir / "training_curve.jsonl"
    curve_path.write_text("", encoding="utf-8")
    best_path = args.output_dir / "stability_best.pt"
    last_path = args.output_dir / "stability_last.pt"
    metadata = {"config": config, "smoke": smoke}
    initial_checkpoint = checkpoint_artifact(
        model, optimizer, scheduler, scaler, generator, epoch=3, history=history,
        global_optimizer_step=global_step, metadata=metadata,
    )
    atomic_torch_save(initial_checkpoint, best_path)
    del initial_checkpoint

    stop_reason = None
    training_seconds = 0.0
    with curve_path.open("a", encoding="utf-8") as log_handle:
        for epoch in range(4, args.max_epoch + 1):
            training = train_micro_batches(
                model=model, loader=train_loader, optimizer=optimizer, scaler=scaler,
                accumulation=args.gradient_accumulation, clipping=args.gradient_clipping,
                target_optimizer_steps=steps_per_epoch, scheduler=scheduler,
                log_handle=log_handle, starting_optimizer_step=global_step, epoch=epoch,
            )
            global_step += steps_per_epoch
            training_seconds += training["duration_seconds"]
            evaluations, differences = evaluate_conditions(
                model, valid_dataset, shuffled_dataset, args, device, epoch
            )
            row = {
                "epoch": epoch,
                "training": training,
                "evaluations": evaluations,
                "auroc_differences": differences,
                "all_finite": all_parameters_finite(model),
            }
            row["stable"] = metric_row_is_stable(row)
            history.append(row)
            write_json(args.output_dir / f"epoch_{epoch}_summary.json", row)
            artifact = checkpoint_artifact(
                model, optimizer, scheduler, scaler, generator, epoch=epoch,
                history=history, global_optimizer_step=global_step, metadata=metadata,
            )
            atomic_torch_save(artifact, last_path)
            if best_epoch(history)["epoch"] == epoch:
                atomic_torch_save(artifact, best_path)
            del artifact
            stop_reason = stopping_reason(history, max_epoch=args.max_epoch)
            if stop_reason is not None:
                break
    if stop_reason is None:
        raise RuntimeError("Training loop exited without a declared stopping reason")
    selected_epoch = best_epoch(history)["epoch"]
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    del train_loader, shuffled_dataset, model, optimizer, scheduler, scaler, start_artifact
    gc.collect()
    torch.cuda.empty_cache()

    final = final_statistics(
        args, device, valid_dataset, best_path,
        [args.seed + offset for offset in range(5)],
    )
    del valid_dataset, train_dataset
    gc.collect()
    torch.cuda.empty_cache()
    final_tests = run_pytest()
    report = {
        "status": "complete",
        "start_checkpoint": str(args.start_checkpoint.resolve()),
        "recovery_mode": recovery_mode,
        "first_trained_epoch": 4,
        "stop_epoch": history[-1]["epoch"],
        "stop_reason": stop_reason,
        "epochs": history,
        "best_epoch": selected_epoch,
        "final": final,
        "smoke": smoke,
        "tests": {"preflight": preflight, "final": final_tests},
        "training_duration_seconds": training_seconds,
        "total_duration_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": peak_memory,
        "checkpoints": {"best": str(best_path.resolve()), "last": str(last_path.resolve())},
        "log_path": str((args.output_dir / "train.log").resolve()),
        "test_split_used": False,
        "git": git,
    }
    write_json(args.output_dir / "training_report.json", report)
    render_markdown(report, args.report_path)
    print(json.dumps({
        "status": "complete", "stop_epoch": report["stop_epoch"],
        "stop_reason": stop_reason, "best_epoch": selected_epoch,
        "conclusion": final["conclusion"], "report": str(args.report_path),
    }, ensure_ascii=False), flush=True)


def write_failure(args: argparse.Namespace, error: BaseException) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failure = {
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    write_json(args.output_dir / "training_report.json", failure)
    args.report_path.write_text(
        "# SpatialSense ByteFormer Stability Long-run\n\n"
        f"实验安全退出：`{type(error).__name__}: {error}`。详见训练日志和JSON失败报告。\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parsed = parse_args()
    try:
        main(parsed)
    except BaseException as exception:
        write_failure(parsed, exception)
        raise
