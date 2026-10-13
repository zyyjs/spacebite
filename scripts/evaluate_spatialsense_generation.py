#!/usr/bin/env python3
"""Evaluate a saved trainable-only bridge on the fixed SpatialSense set."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from spacebyte.data import SpatialSenseCachedFeatureDataset, parse_binary_answer
from spacebyte.models import (
    ByteFormerTokenAdapter,
    ByteQFormerBridge,
    ByteVicuna,
    build_official_byteformer,
    build_xinstruct_qformer_tokenizer,
    load_frozen_vicuna,
)
from spacebyte.training import supervised_token_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--bridge-checkpoint", type=Path, required=True)
    parser.add_argument("--vicuna-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--qformer-tokenizer", type=Path, required=True)
    parser.add_argument("--byteformer-config", type=Path, required=True)
    parser.add_argument("--byteformer-checkpoint", type=Path, required=True)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dataset = SpatialSenseCachedFeatureDataset(args.manifest, args.cache_root)
    bridge = ByteQFormerBridge().to(device)
    saved = torch.load(args.bridge_checkpoint, map_location="cpu", weights_only=True)
    incompatible = bridge.load_state_dict(saved["bridge_state_dict"], strict=True)
    byteformer, _, byteformer_load = build_official_byteformer(
        args.byteformer_config, args.byteformer_checkpoint
    )
    adapter = ByteFormerTokenAdapter(byteformer).to(device)
    vicuna, llm_tokenizer, vicuna_report = load_frozen_vicuna(
        args.vicuna_path, device=device, dtype=torch.float16
    )
    model = ByteVicuna(
        byteformer_adapter=adapter,
        bridge=bridge,
        vicuna=vicuna,
        llm_tokenizer=llm_tokenizer,
        qformer_tokenizer=build_xinstruct_qformer_tokenizer(
            args.qformer_tokenizer
        ),
    ).eval()
    total_loss = 0.0
    exact = parsed = token_correct = token_total = 0
    distribution: Counter[str] = Counter()
    relation_counts: dict[str, Counter[str]] = defaultdict(Counter)
    outputs = []
    for index in range(len(dataset)):
        sample = dataset[index]
        features = sample["byte_features"].unsqueeze(0).to(device)
        mask = sample["byte_attention_mask"].unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            result = model.forward_from_byte_features(
                features, mask, sample["question"], sample["answer"]
            )
            generated, ids, _ = model.generate_from_byte_features(
                features, mask, sample["question"], num_beams=1, max_length=5
            )
        prediction = parse_binary_answer(generated[0])
        key = prediction or "<unparsed>"
        correct = prediction == sample["answer"]
        exact += int(correct)
        parsed += int(prediction is not None)
        distribution[key] += 1
        relation_counts[sample["relation"]]["total"] += 1
        relation_counts[sample["relation"]]["correct"] += int(correct)
        relation_counts[sample["relation"]][key] += 1
        row_correct, row_total = supervised_token_accuracy(
            result.logits, result.prepared.labels
        )
        token_correct += row_correct
        token_total += row_total
        total_loss += float(result.loss.float().item())
        outputs.append(
            {
                "sample_id": sample["sample_id"],
                "answer": sample["answer"],
                "generated": generated[0],
                "token_ids": ids[0].tolist(),
                "parsed": prediction,
                "correct": correct,
            }
        )
    report = {
        "bridge_checkpoint": str(args.bridge_checkpoint.resolve()),
        "bridge_checkpoint_step": saved["step"],
        "strict_load_missing_keys": list(incompatible.missing_keys),
        "strict_load_unexpected_keys": list(incompatible.unexpected_keys),
        "byteformer_checkpoint": byteformer_load,
        "vicuna": vicuna_report,
        "sample_count": len(dataset),
        "mean_loss": total_loss / len(dataset),
        "exact_match_accuracy": exact / len(dataset),
        "parse_rate": parsed / len(dataset),
        "token_accuracy": token_correct / token_total,
        "prediction_distribution": dict(distribution),
        "per_relation": {
            relation: {
                **dict(counts),
                "accuracy": counts["correct"] / counts["total"],
            }
            for relation, counts in sorted(relation_counts.items())
        },
        "outputs": outputs,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
