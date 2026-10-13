#!/usr/bin/env python3
"""Run the fixed SpatialSense candidate-relation tiny overfit experiment."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from spacebyte.data import SpatialSenseCachedFeatureDataset, load_json, parse_binary_answer
from spacebyte.data.spatialsense import render_record_to_jpeg_bytes
from spacebyte.models import (
    ByteFormerTokenAdapter,
    ByteQFormerBridge,
    ByteVicuna,
    build_official_byteformer,
    build_xinstruct_qformer_tokenizer,
    initialize_byte_qformer_from_blip2_stage1,
    load_frozen_vicuna,
)
from spacebyte.training import (
    assert_frozen_modules_absent,
    audit_causal_labels,
    build_bridge_optimizer_groups,
    supervised_token_accuracy,
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
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--vicuna-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--qformer-tokenizer", type=Path, default=DEFAULT_QFORMER_TOKENIZER)
    parser.add_argument("--blip2-checkpoint", type=Path, default=DEFAULT_BLIP2)
    parser.add_argument("--byteformer-config", type=Path, default=DEFAULT_BYTEFORMER_CONFIG)
    parser.add_argument("--byteformer-checkpoint", type=Path, default=DEFAULT_BYTEFORMER_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--pretrained-lr", type=float, default=1e-5)
    parser.add_argument("--new-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clipping", type=float, default=1.0)
    parser.add_argument("--success-exact-match", type=float, default=0.90)
    parser.add_argument("--success-patience", type=int, default=2)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cached_sample(dataset, index, device):
    sample = dataset[index]
    return sample, sample["byte_features"].unsqueeze(0).to(device), sample[
        "byte_attention_mask"
    ].unsqueeze(0).to(device)


@torch.no_grad()
def evaluate(model: ByteVicuna, dataset, device: torch.device) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    token_correct = 0
    token_total = 0
    exact = 0
    parsed = 0
    prediction_counts: Counter[str] = Counter()
    per_relation: dict[str, Counter[str]] = defaultdict(Counter)
    outputs = []
    for index in range(len(dataset)):
        sample, features, mask = cached_sample(dataset, index, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model.forward_from_byte_features(
                features, mask, sample["question"], sample["answer"]
            )
            generated, token_ids, _ = model.generate_from_byte_features(
                features,
                mask,
                sample["question"],
                num_beams=1,
                max_length=5,
            )
        prediction = parse_binary_answer(generated[0])
        prediction_key = prediction or "<unparsed>"
        is_correct = prediction == sample["answer"]
        exact += int(is_correct)
        parsed += int(prediction is not None)
        prediction_counts[prediction_key] += 1
        per_relation[sample["relation"]]["total"] += 1
        per_relation[sample["relation"]]["correct"] += int(is_correct)
        per_relation[sample["relation"]][prediction_key] += 1
        correct, count = supervised_token_accuracy(
            output.logits, output.prepared.labels
        )
        token_correct += correct
        token_total += count
        total_loss += float(output.loss.float().item())
        outputs.append(
            {
                "sample_id": sample["sample_id"],
                "relation": sample["relation"],
                "answer": sample["answer"],
                "generated": generated[0],
                "generated_token_ids": token_ids[0].tolist(),
                "parsed": prediction,
                "exact_match": is_correct,
            }
        )
    return {
        "mean_loss": total_loss / len(dataset),
        "exact_match_accuracy": exact / len(dataset),
        "exact_match_count": exact,
        "sample_count": len(dataset),
        "parse_rate": parsed / len(dataset),
        "token_accuracy": token_correct / max(token_total, 1),
        "token_correct": token_correct,
        "token_total": token_total,
        "prediction_distribution": dict(prediction_counts),
        "per_relation": {
            relation: {
                **dict(counts),
                "accuracy": counts["correct"] / counts["total"],
            }
            for relation, counts in sorted(per_relation.items())
        },
        "outputs": outputs,
    }


def parameter_versions(module) -> dict[str, int]:
    return {name: parameter._version for name, parameter in module.named_parameters()}


def representative_parameters(bridge) -> dict[str, torch.Tensor]:
    wanted = {
        "query_tokens",
        "byte_layernorm.weight",
        "qformer.bert.encoder.layer.0.attention.self.query.weight",
        "qformer.bert.encoder.layer.0.crossattention.self.key.weight",
        "qformer.bert.encoder.layer.2.crossattention.self.key.weight",
        "qformer.bert.encoder.layer.4.crossattention.self.key.weight",
        "qformer.bert.encoder.layer.6.crossattention.self.key.weight",
        "qformer.bert.encoder.layer.8.crossattention.self.key.weight",
        "qformer.bert.encoder.layer.10.crossattention.self.key.weight",
        "llm_projector.weight",
    }
    named = dict(bridge.named_parameters())
    missing = wanted - set(named)
    if missing:
        raise RuntimeError(f"Representative update audit keys missing: {sorted(missing)}")
    return {name: named[name].detach().cpu().clone() for name in sorted(wanted)}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("This FP16 overfit verification requires CUDA")
    set_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_json(args.manifest)
    if manifest["seed"] != args.seed:
        raise RuntimeError("Training seed must match the committed tiny manifest")
    dataset = SpatialSenseCachedFeatureDataset(args.manifest, args.cache_root)

    bridge = ByteQFormerBridge().to(device)
    initialization = initialize_byte_qformer_from_blip2_stage1(
        bridge, args.blip2_checkpoint
    )
    byteformer, _, byteformer_load = build_official_byteformer(
        args.byteformer_config, args.byteformer_checkpoint
    )
    adapter = ByteFormerTokenAdapter(byteformer).to(device)
    vicuna, llm_tokenizer, vicuna_report = load_frozen_vicuna(
        args.vicuna_path, device=device, dtype=torch.float16
    )
    qformer_tokenizer = build_xinstruct_qformer_tokenizer(args.qformer_tokenizer)
    model = ByteVicuna(
        byteformer_adapter=adapter,
        bridge=bridge,
        vicuna=vicuna,
        llm_tokenizer=llm_tokenizer,
        qformer_tokenizer=qformer_tokenizer,
    ).to(device)
    memory_after_load = int(torch.cuda.memory_allocated(device))

    optimizer_groups, grouping_report = build_bridge_optimizer_groups(
        bridge,
        pretrained_lr=args.pretrained_lr,
        new_lr=args.new_lr,
        weight_decay=args.weight_decay,
    )
    assert_frozen_modules_absent(
        optimizer_groups, model.byteformer_adapter, model.vicuna
    )
    optimizer = torch.optim.AdamW(optimizer_groups)
    scaler = torch.cuda.amp.GradScaler()

    frozen_versions_before = {
        "byteformer": parameter_versions(model.byteformer_adapter),
        "vicuna": parameter_versions(model.vicuna),
    }
    trainable_before = representative_parameters(bridge)

    first_record = manifest["samples"][0]
    online_jpeg, _, _ = render_record_to_jpeg_bytes(args.dataset_root, first_record)
    online_jpeg = online_jpeg.unsqueeze(0).to(device)
    model.eval()
    # Keep the official frozen ByteFormer path in FP32, matching cache creation.
    # ByteVicuna converts only the projected Query prefix to Vicuna's FP16 dtype.
    with torch.no_grad():
        online_output = model(
            online_jpeg, first_record["question"], first_record["answer"]
        )
    online_label_audit = audit_causal_labels(
        online_output.prepared, [first_record["answer"]], llm_tokenizer
    )
    cached_first = dataset[0]
    online_valid = online_output.prepared.byte_attention_mask[0].bool()
    online_features = online_output.prepared.byte_features[0][online_valid].float().cpu()
    cached_features = cached_first["byte_features"][
        cached_first["byte_attention_mask"].bool()
    ].float()
    if online_features.shape != cached_features.shape:
        raise RuntimeError("Online and cached ByteFormer feature shapes differ")
    online_cache_error = (online_features - cached_features).abs()
    end_to_end_report = {
        "sample_id": first_record["sample_id"],
        "jpeg_byte_length": online_jpeg.shape[1],
        "loss": float(online_output.loss.float().item()),
        "finite": bool(torch.isfinite(online_output.loss).item()),
        "byte_feature_shape": list(online_output.prepared.byte_features.shape),
        "query_shape": list(online_output.prepared.query_features.shape),
        "projected_shape": list(online_output.prepared.projected_queries.shape),
        "max_cache_absolute_error": float(online_cache_error.max().item()),
        "mean_cache_absolute_error": float(online_cache_error.mean().item()),
        "label_audit": online_label_audit,
    }

    start_time = time.perf_counter()
    initial = evaluate(model, dataset, device)
    history = [
        {
            "step": 0,
            "elapsed_seconds": time.perf_counter() - start_time,
            "samples_per_second": 0.0,
            "gradient_norm": None,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "evaluation": initial,
        }
    ]
    log_path = args.output_dir / "training_log.jsonl"
    log_path.write_text(json.dumps(history[0], ensure_ascii=False) + "\n", encoding="utf-8")

    rng = random.Random(args.seed)
    order = list(range(len(dataset)))
    rng.shuffle(order)
    order_position = 0
    optimizer.zero_grad(set_to_none=True)
    successful_evaluations = 0
    final = initial
    completed_steps = 0
    accumulated_loss = 0.0
    interval_start = time.perf_counter()
    interval_samples = 0
    last_gradient_norm = 0.0
    for step in range(1, args.max_steps + 1):
        model.train()
        for _ in range(args.gradient_accumulation):
            if order_position == len(order):
                rng.shuffle(order)
                order_position = 0
            index = order[order_position]
            order_position += 1
            sample, features, mask = cached_sample(dataset, index, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model.forward_from_byte_features(
                    features, mask, sample["question"], sample["answer"]
                )
                if not torch.isfinite(output.loss):
                    raise RuntimeError(f"Non-finite loss at optimizer step {step}")
                scaled_loss = output.loss / args.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            accumulated_loss += float(output.loss.detach().float().item())
            interval_samples += 1
        scaler.unscale_(optimizer)
        last_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                bridge.parameters(), args.gradient_clipping
            ).item()
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        completed_steps = step

        if step % args.eval_interval == 0 or step == args.max_steps:
            evaluation = evaluate(model, dataset, device)
            now = time.perf_counter()
            row = {
                "step": step,
                "mean_training_loss_since_last_eval": accumulated_loss / interval_samples,
                "elapsed_seconds": now - start_time,
                "samples_per_second": interval_samples / (now - interval_start),
                "gradient_norm": last_gradient_norm,
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                "evaluation": evaluation,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            history.append(row)
            final = evaluation
            accumulated_loss = 0.0
            interval_samples = 0
            interval_start = now
            successful_evaluations = (
                successful_evaluations + 1
                if evaluation["exact_match_accuracy"] >= args.success_exact_match
                else 0
            )
            print(
                f"step={step} loss={evaluation['mean_loss']:.6f} "
                f"exact={evaluation['exact_match_accuracy']:.3f} "
                f"token={evaluation['token_accuracy']:.3f}",
                flush=True,
            )
            if successful_evaluations >= args.success_patience:
                break

    duration = time.perf_counter() - start_time
    trainable_after = dict(bridge.named_parameters())
    update_report = {
        name: {
            "max_absolute_change": float(
                (trainable_after[name].detach().float().cpu() - before.float())
                .abs()
                .max()
                .item()
            )
        }
        for name, before in trainable_before.items()
    }
    if not all(row["max_absolute_change"] > 0 for row in update_report.values()):
        raise RuntimeError(f"Representative trainable parameter did not update: {update_report}")

    frozen_versions_after = {
        "byteformer": parameter_versions(model.byteformer_adapter),
        "vicuna": parameter_versions(model.vicuna),
    }
    frozen_report = {
        name: {
            "all_parameter_versions_unchanged": frozen_versions_before[name]
            == frozen_versions_after[name],
            "parameters_with_grad": [
                parameter_name
                for parameter_name, parameter in module.named_parameters()
                if parameter.grad is not None
            ],
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in module.parameters() if parameter.requires_grad
            ),
        }
        for name, module in (
            ("byteformer", model.byteformer_adapter),
            ("vicuna", model.vicuna),
        )
    }
    if not all(
        row["all_parameter_versions_unchanged"]
        and not row["parameters_with_grad"]
        and row["trainable_parameter_count"] == 0
        for row in frozen_report.values()
    ):
        raise RuntimeError(f"Frozen module audit failed: {frozen_report}")

    bridge_path = args.output_dir / "bridge_final.pt"
    optimizer_path = args.output_dir / "optimizer_final.pt"
    torch.save(
        {
            "bridge_state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in bridge.state_dict().items()
            },
            "step": completed_steps,
            "manifest": str(args.manifest.resolve()),
            "seed": args.seed,
        },
        bridge_path,
    )
    torch.save(
        {"optimizer_state_dict": optimizer.state_dict(), "step": completed_steps},
        optimizer_path,
    )
    configuration = {
        "seed": args.seed,
        "precision": (
            "frozen ByteFormer FP32; FP16 feature cache; bridge/Vicuna training "
            "under FP16 autocast with FP32 trainable parameters and GradScaler"
        ),
        "micro_batch_size": 1,
        "gradient_accumulation": args.gradient_accumulation,
        "max_steps": args.max_steps,
        "completed_steps": completed_steps,
        "optimizer": "AdamW",
        "gradient_clipping": args.gradient_clipping,
        "use_cache": False,
        "pretrained_lr": args.pretrained_lr,
        "new_lr": args.new_lr,
        "weight_decay": args.weight_decay,
        "eval_interval": args.eval_interval,
        "quantization": False,
        "lora": False,
        "crema": False,
        "multi_gpu": False,
        "gradient_checkpointing": False,
    }
    report = {
        "task": "SpatialSense candidate-relation yes/no tiny overfit",
        "configuration": configuration,
        "manifest": str(args.manifest.resolve()),
        "cache_root": str(args.cache_root.resolve()),
        "sample_count": len(dataset),
        "byteformer_checkpoint": byteformer_load,
        "blip2_initialization": initialization["summary"],
        "vicuna": vicuna_report,
        "optimizer_grouping": grouping_report,
        "online_end_to_end_before_training": end_to_end_report,
        "initial": initial,
        "final": final,
        "history": history,
        "trainable_parameter_updates": update_report,
        "frozen_modules": frozen_report,
        "duration_seconds": duration,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "memory_after_model_load_bytes": memory_after_load,
        "success": final["exact_match_accuracy"] >= args.success_exact_match,
        "bridge_checkpoint": str(bridge_path.resolve()),
        "optimizer_checkpoint": str(optimizer_path.resolve()),
    }
    (args.output_dir / "training_config.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
