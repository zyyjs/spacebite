#!/usr/bin/env python3
"""Run exactly one formal SpatialSense Byte-Vicuna pilot epoch."""

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

from spacebyte.data import (
    SpatialSenseFullCachedDataset,
    atomic_torch_save,
    collate_cached_features,
    sha256_file,
)
from spacebyte.evaluation import fixed_derangement
from spacebyte.full_pipeline import (
    PermutedVisualDataset,
    build_full_model,
    evaluate_full,
)
from spacebyte.training import (
    assert_frozen_modules_absent,
    audit_causal_labels,
    build_bridge_optimizer_groups,
    checkpoint_is_better,
    gradient_norm_for_names,
    warmup_cosine_multiplier,
)


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


def git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], text=True, cwd=Path(__file__).resolve().parents[1]
        ).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_short": run("status", "--short"),
    }


def bridge_artifact(model, *, step: int, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": 1,
        "bridge_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.bridge.state_dict().items()
        },
        "optimizer_step": step,
        "metadata": metadata,
    }


def evaluation_summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in evaluation.items() if key != "outputs"}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def gradient_report(bridge) -> dict[str, float]:
    cross_kv = lambda name: (
        ".crossattention.self.key.weight" in name
        or ".crossattention.self.value.weight" in name
    )
    return {
        "cross_attention_kv": gradient_norm_for_names(bridge, cross_kv),
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
        raise RuntimeError("The formal FP16 pilot requires a CUDA device")
    if args.gradient_accumulation != 8:
        raise ValueError("The fixed formal pilot requires gradient accumulation 8")
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
    if len(train_dataset) != 11238 or len(valid_dataset) != 2638:
        raise RuntimeError("Official train/valid annotation counts changed")
    if {row["sample_id"] for row in train_dataset.records} & {
        row["sample_id"] for row in valid_dataset.records
    }:
        raise RuntimeError("Train/valid sample IDs overlap")

    git = git_metadata()
    model, load_report = build_full_model(
        device=device,
        byteformer_config=args.byteformer_config,
        byteformer_checkpoint=args.byteformer_checkpoint,
        blip2_checkpoint=args.blip2_checkpoint,
        vicuna_path=args.vicuna_path,
        qformer_tokenizer_path=args.qformer_tokenizer,
    )
    memory_after_model_load = int(torch.cuda.memory_allocated(device))
    optimizer_groups, optimizer_group_report = build_bridge_optimizer_groups(
        model.bridge,
        pretrained_lr=args.pretrained_lr,
        new_lr=args.new_lr,
        weight_decay=args.weight_decay,
    )
    assert_frozen_modules_absent(
        optimizer_groups, model.byteformer_adapter, model.vicuna
    )
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

    answer_tokenization = {}
    for answer in ("yes", "no"):
        raw = model.llm_tokenizer(
            f"{answer}{model.llm_tokenizer.eos_token}", add_special_tokens=True
        ).input_ids
        supervised = list(raw[1:])
        decoded = model.llm_tokenizer.decode(
            supervised, skip_special_tokens=False
        )
        answer_tokenization[answer] = {
            "raw_with_start_token_ids": raw,
            "supervised_answer_plus_eos_ids": supervised,
            "tokens": model.llm_tokenizer.convert_ids_to_tokens(supervised),
            "decoded": decoded,
            "contains_leading_space": decoded.startswith(" "),
            "supervised_token_count": len(supervised),
        }

    audit_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_cached_features,
    )
    audit_batch = next(iter(audit_loader))
    audit_features = audit_batch["byte_features"].to(device)
    audit_mask = audit_batch["byte_attention_mask"].to(device)
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        audit_output = model.forward_from_byte_features(
            audit_features,
            audit_mask,
            audit_batch["questions"],
            audit_batch["answers"],
        )
    label_audit = audit_causal_labels(
        audit_output.prepared, audit_batch["answers"], model.llm_tokenizer
    )

    cache_manifest_sha = {
        "train": sha256_file(args.train_manifest),
        "valid": sha256_file(args.valid_manifest),
    }
    config = {
        "task": "Byte-stream-conditioned spatial relation verification",
        "epochs": 1,
        "micro_batch_size": 1,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": args.gradient_accumulation,
        "optimizer_steps": optimizer_steps,
        "precision": "FP16 autocast with GradScaler",
        "gradient_scaler_initial_scale": args.grad_scaler_init,
        "gradient_scaler_growth_interval": 2000,
        "optimizer": "AdamW",
        "gradient_clipping": args.gradient_clipping,
        "seed": args.seed,
        "pretrained_lr": args.pretrained_lr,
        "new_lr": args.new_lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "scheduler": "linear warmup then cosine decay",
        "byteformer_frozen": True,
        "vicuna_frozen": True,
        "use_cache": False,
        "train_manifest": str(args.train_manifest.resolve()),
        "valid_manifest": str(args.valid_manifest.resolve()),
        "cache_manifest_sha256": cache_manifest_sha,
        "git_at_launch": git,
        "answer_tokenization": answer_tokenization,
        "label_audit": label_audit,
        "optimizer_groups": optimizer_group_report,
        "model_load": load_report,
        "test_split_used": False,
    }
    write_json(args.output_dir / "training_config.json", config)

    checkpoint_metadata = {
        "training_config": config,
        "random_seed": args.seed,
        "git_commit": git["commit"],
        "cache_manifest_sha256": cache_manifest_sha,
    }
    validation_steps = {0, math.ceil(optimizer_steps / 2), optimizer_steps}
    best_accuracy = None
    best_auroc = None
    best_step = None
    validations = {}

    def run_validation(step: int) -> dict[str, Any]:
        nonlocal best_accuracy, best_auroc, best_step
        print(f"starting full validation at optimizer step {step}", flush=True)
        evaluation = evaluate_full(
            model,
            valid_dataset,
            device=device,
            batch_size=args.eval_batch_size,
            projected_query_scale=1.0,
            num_workers=args.num_workers,
        )
        path = args.output_dir / f"validation_step_{step}.json"
        write_json(path, evaluation)
        summary = evaluation_summary(evaluation)
        validations[str(step)] = {"path": str(path.resolve()), **summary}
        metric = evaluation["likelihood"]
        if checkpoint_is_better(
            metric["accuracy"], metric["auroc"], best_accuracy, best_auroc
        ):
            best_accuracy = metric["accuracy"]
            best_auroc = metric["auroc"]
            best_step = step
            atomic_torch_save(
                bridge_artifact(
                    model,
                    step=step,
                    metadata={
                        **checkpoint_metadata,
                        "selection_metric": {
                            "likelihood_accuracy": best_accuracy,
                            "likelihood_auroc": best_auroc,
                        },
                    },
                ),
                args.output_dir / "bridge_best.pt",
            )
        print(
            f"validation step={step} greedy={evaluation['greedy']['exact_match_accuracy']:.4f} "
            f"likelihood={metric['accuracy']:.4f} auroc={metric['auroc']:.4f}",
            flush=True,
        )
        return evaluation

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
    log_path = args.output_dir / "training_curve.jsonl"
    log_path.write_text("", encoding="utf-8")
    optimizer.zero_grad(set_to_none=True)
    model.train()
    micro_step = 0
    optimizer_step = 0
    accumulated_loss = 0.0
    accumulated_samples = 0
    interval_started = time.perf_counter()
    nonfinite_events = 0
    training_compute_seconds = 0.0
    epoch_started = time.perf_counter()

    for batch in train_loader:
        group_start = (micro_step // args.gradient_accumulation) * args.gradient_accumulation
        group_size = min(
            args.gradient_accumulation, len(train_dataset) - group_start
        )
        features = batch["byte_features"].to(device, non_blocking=True)
        mask = batch["byte_attention_mask"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model.forward_from_byte_features(
                features,
                mask,
                batch["questions"],
                batch["answers"],
            )
            if not bool(torch.isfinite(output.loss).item()):
                nonfinite_events += 1
                raise RuntimeError(f"Non-finite loss at micro step {micro_step}")
            loss = output.loss / group_size
        scaler.scale(loss).backward()
        accumulated_loss += float(output.loss.detach().float().item())
        accumulated_samples += 1
        micro_step += 1
        group_complete = (
            micro_step % args.gradient_accumulation == 0
            or micro_step == len(train_dataset)
        )
        if not group_complete:
            continue

        scaler.unscale_(optimizer)
        gradients = gradient_report(model.bridge)
        total_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.bridge.parameters(), args.gradient_clipping
            ).item()
        )
        if not math.isfinite(total_gradient_norm) or not all(
            math.isfinite(value) for value in gradients.values()
        ):
            nonfinite_events += 1
            raise RuntimeError(f"Non-finite gradient at optimizer step {optimizer_step + 1}")
        learning_rates_used = {
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
            "learning_rates_used": learning_rates_used,
            "learning_rates_next_step": {
                group["group_name"]: group["lr"]
                for group in optimizer.param_groups
            },
            "total_gradient_norm_before_clipping": total_gradient_norm,
            "gradient_norms": gradients,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "samples_per_second": accumulated_samples
            / max(now - interval_started, 1e-12),
            "elapsed_seconds": now - pilot_started,
            "nonfinite_events": nonfinite_events,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if optimizer_step % 25 == 0 or optimizer_step in validation_steps:
            print(
                f"step={optimizer_step}/{optimizer_steps} loss={row['mean_loss']:.5f} "
                f"grad={total_gradient_norm:.4f} samples/s={row['samples_per_second']:.2f}",
                flush=True,
            )
        accumulated_loss = 0.0
        accumulated_samples = 0
        interval_started = now
        if optimizer_step in validation_steps:
            run_validation(optimizer_step)
            model.train()
            interval_started = time.perf_counter()

    if optimizer_step != optimizer_steps or micro_step != len(train_dataset):
        raise RuntimeError("One-epoch step accounting failed")
    epoch_seconds = time.perf_counter() - epoch_started

    atomic_torch_save(
        bridge_artifact(
            model,
            step=optimizer_step,
            metadata={**checkpoint_metadata, "kind": "last"},
        ),
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

    valid_visual_keys = [
        row["visual_cache_key"] for row in valid_dataset.records
    ]
    permutation = fixed_derangement(
        len(valid_dataset), args.seed, valid_visual_keys
    )
    permutation_rows = [
        {
            "target_index": index,
            "target_sample_id": valid_dataset.records[index]["sample_id"],
            "source_index": source,
            "source_sample_id": valid_dataset.records[source]["sample_id"],
            "target_visual_cache_key": valid_visual_keys[index],
            "source_visual_cache_key": valid_visual_keys[source],
        }
        for index, source in enumerate(permutation)
    ]
    permutation_path = args.output_dir / "valid_shuffled_byte_mapping.json"
    write_json(
        permutation_path,
        {"seed": args.seed, "mapping": permutation_rows},
    )
    print("starting epoch-end Zero-Byte validation", flush=True)
    zero_evaluation = evaluate_full(
        model,
        valid_dataset,
        device=device,
        batch_size=args.eval_batch_size,
        projected_query_scale=0.0,
        num_workers=args.num_workers,
    )
    write_json(args.output_dir / "validation_zero_byte.json", zero_evaluation)
    print("starting epoch-end Shuffled-Byte validation", flush=True)
    shuffled_evaluation = evaluate_full(
        model,
        PermutedVisualDataset(valid_dataset, permutation),
        device=device,
        batch_size=args.eval_batch_size,
        projected_query_scale=1.0,
        num_workers=args.num_workers,
    )
    write_json(
        args.output_dir / "validation_shuffled_byte.json", shuffled_evaluation
    )

    frozen_report = {}
    for name, module in (
        ("byteformer", model.byteformer_adapter),
        ("vicuna", model.vicuna),
    ):
        frozen_report[name] = {
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in module.parameters()
                if parameter.requires_grad
            ),
            "parameters_with_grad": [
                parameter_name
                for parameter_name, parameter in module.named_parameters()
                if parameter.grad is not None
            ],
        }
    if any(
        row["trainable_parameter_count"] or row["parameters_with_grad"]
        for row in frozen_report.values()
    ):
        raise RuntimeError(f"Frozen module audit failed: {frozen_report}")

    normal_final = validations[str(optimizer_steps)]
    comparison = {
        "normal": {
            "greedy": normal_final["greedy"],
            "likelihood": normal_final["likelihood"],
        },
        "zero_byte": {
            "greedy": zero_evaluation["greedy"],
            "likelihood": zero_evaluation["likelihood"],
        },
        "shuffled_byte": {
            "greedy": shuffled_evaluation["greedy"],
            "likelihood": shuffled_evaluation["likelihood"],
        },
    }
    normal_accuracy = comparison["normal"]["likelihood"]["accuracy"]
    largest_control = max(
        comparison["zero_byte"]["likelihood"]["accuracy"],
        comparison["shuffled_byte"]["likelihood"]["accuracy"],
    )
    comparison["normal_minus_zero_likelihood_accuracy"] = normal_accuracy - comparison[
        "zero_byte"
    ]["likelihood"]["accuracy"]
    comparison["normal_minus_shuffled_likelihood_accuracy"] = normal_accuracy - comparison[
        "shuffled_byte"
    ]["likelihood"]["accuracy"]
    comparison["interpretation"] = (
        "Normal likelihood accuracy exceeds both controls by at least 2 percentage points."
        if normal_accuracy - largest_control >= 0.02
        else "Normal does not clearly exceed both controls; this pilot may rely mainly on relation text and yes/no priors."
    )
    write_json(args.output_dir / "epoch_end_ablation_comparison.json", comparison)

    report = {
        "task": config["task"],
        "config": config,
        "answer_tokenization": answer_tokenization,
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
        "best_checkpoint_step": best_step,
        "best_likelihood_accuracy": best_accuracy,
        "best_likelihood_auroc": best_auroc,
        "epoch_end_ablation": comparison,
        "permutation_manifest": str(permutation_path.resolve()),
        "frozen_modules": frozen_report,
        "artifacts": {
            "best_bridge": str((args.output_dir / "bridge_best.pt").resolve()),
            "last_bridge": str((args.output_dir / "bridge_last.pt").resolve()),
            "optimizer_scheduler": str(
                (args.output_dir / "optimizer_scheduler_last.pt").resolve()
            ),
            "training_curve": str(log_path.resolve()),
        },
        "test_split_used": False,
    }
    write_json(args.output_dir / "training_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
