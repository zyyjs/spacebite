"""Reusable construction and evaluation for the full SpatialSense pilot."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .data import collate_cached_features
from .evaluation import (
    conditional_log_likelihood,
    greedy_metrics,
    likelihood_metrics,
    parse_greedy_binary,
)
from .models import (
    ByteFormerTokenAdapter,
    ByteQFormerBridge,
    ByteVicuna,
    build_official_byteformer,
    build_xinstruct_qformer_tokenizer,
    initialize_byte_qformer_from_blip2_stage1,
    load_frozen_vicuna,
)


class PermutedVisualDataset(Dataset):
    """Retain target text/label while sourcing visual tensors by fixed index."""

    def __init__(self, dataset: Dataset, mapping: list[int]) -> None:
        if len(dataset) != len(mapping):
            raise ValueError("Permutation length does not match dataset")
        if sorted(mapping) != list(range(len(dataset))):
            raise ValueError("Visual mapping is not a permutation")
        if any(index == source for index, source in enumerate(mapping)):
            raise ValueError("Visual permutation contains a self mapping")
        records = getattr(dataset, "records", None)
        if records is not None and any(
            records[index].get("visual_cache_key")
            == records[source].get("visual_cache_key")
            for index, source in enumerate(mapping)
        ):
            raise ValueError("Visual permutation reuses a target visual identity")
        self.dataset = dataset
        self.mapping = mapping

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        target = dict(self.dataset[index])
        source = self.dataset[self.mapping[index]]
        target["byte_features"] = source["byte_features"]
        target["byte_attention_mask"] = source["byte_attention_mask"]
        target["source_visual_cache_key"] = source.get("visual_cache_key")
        return target


def build_full_model(
    *,
    device: torch.device,
    byteformer_config: Path,
    byteformer_checkpoint: Path,
    blip2_checkpoint: Path,
    vicuna_path: Path,
    qformer_tokenizer_path: Path,
    bridge_checkpoint: Path | None = None,
) -> tuple[ByteVicuna, dict[str, Any]]:
    bridge = ByteQFormerBridge()
    initialization = initialize_byte_qformer_from_blip2_stage1(
        bridge, blip2_checkpoint
    )
    bridge_load = None
    if bridge_checkpoint is not None:
        artifact = torch.load(
            bridge_checkpoint, map_location="cpu", weights_only=False
        )
        state = artifact.get("bridge_state_dict", artifact)
        incompatible = bridge.load_state_dict(state, strict=True)
        bridge_load = {
            "path": str(bridge_checkpoint.resolve()),
            "strict": True,
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }
    bridge.to(device)
    byteformer, _, byteformer_load = build_official_byteformer(
        byteformer_config, byteformer_checkpoint
    )
    adapter = ByteFormerTokenAdapter(byteformer).to(device).eval()
    vicuna, llm_tokenizer, vicuna_report = load_frozen_vicuna(
        vicuna_path, device=device, dtype=torch.float16
    )
    qformer_tokenizer = build_xinstruct_qformer_tokenizer(
        qformer_tokenizer_path
    )
    model = ByteVicuna(
        byteformer_adapter=adapter,
        bridge=bridge,
        vicuna=vicuna,
        llm_tokenizer=llm_tokenizer,
        qformer_tokenizer=qformer_tokenizer,
    ).to(device)
    return model, {
        "blip2_initialization": initialization["summary"],
        "bridge_checkpoint_load": bridge_load,
        "byteformer_checkpoint_load": byteformer_load,
        "vicuna": vicuna_report,
    }


@torch.no_grad()
def evaluate_full(
    model: ByteVicuna,
    dataset: Dataset,
    *,
    device: torch.device,
    batch_size: int,
    projected_query_scale: float = 1.0,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Run strict greedy generation and length-normalized answer likelihood."""

    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_cached_features,
        pin_memory=num_workers > 0,
    )
    labels: list[bool] = []
    relations: list[str] = []
    scores: list[float] = []
    greedy_predictions: list[str] = []
    outputs: list[dict[str, Any]] = []
    candidate_lengths: set[tuple[int, int]] = set()
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        features = batch["byte_features"].to(device, non_blocking=True)
        mask = batch["byte_attention_mask"].to(device, non_blocking=True)
        size = features.shape[0]
        candidate_features = features.repeat_interleave(2, dim=0)
        candidate_mask = mask.repeat_interleave(2, dim=0)
        candidate_questions = [
            question
            for question in batch["questions"]
            for _ in (0, 1)
        ]
        candidate_answers = [
            answer for _ in range(size) for answer in ("yes", "no")
        ]
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            candidate_output = model.forward_from_byte_features(
                candidate_features,
                candidate_mask,
                candidate_questions,
                candidate_answers,
                projected_query_scale=projected_query_scale,
            )
            generated, generated_ids, _ = model.generate_from_byte_features(
                features,
                mask,
                batch["questions"],
                do_sample=False,
                num_beams=1,
                max_length=5,
                min_length=1,
                projected_query_scale=projected_query_scale,
            )
        sums, counts = conditional_log_likelihood(
            candidate_output.logits, candidate_output.prepared.labels
        )
        # Explicit mean log probability prevents token-length preference. The
        # current yes/no+EOS sequences happen to have equal length.
        normalized = sums / counts
        normalized = normalized.view(size, 2)
        counts = counts.view(size, 2)
        batch_scores = normalized[:, 0] - normalized[:, 1]
        parsed = [parse_greedy_binary(text) for text in generated]
        labels.extend(batch["labels"])
        relations.extend(batch["relations_raw"])
        scores.extend(batch_scores.float().cpu().tolist())
        greedy_predictions.extend(parsed)
        for index in range(size):
            candidate_lengths.add(
                (int(counts[index, 0].item()), int(counts[index, 1].item()))
            )
            outputs.append(
                {
                    "sample_id": batch["sample_ids"][index],
                    "relation": batch["relations_raw"][index],
                    "label": batch["labels"][index],
                    "generated": generated[index],
                    "generated_token_ids": generated_ids[index].cpu().tolist(),
                    "greedy_prediction": parsed[index],
                    "yes_mean_log_probability": float(
                        normalized[index, 0].item()
                    ),
                    "no_mean_log_probability": float(
                        normalized[index, 1].item()
                    ),
                    "score": float(batch_scores[index].item()),
                }
            )
        if (batch_index + 1) % 100 == 0:
            print(
                f"evaluation batches={batch_index + 1}/{len(loader)}",
                flush=True,
            )
    return {
        "sample_count": len(labels),
        "projected_query_scale": projected_query_scale,
        "duration_seconds": time.perf_counter() - started,
        "likelihood_length_normalization": "mean_log_probability_per_supervised_token",
        "candidate_token_length_pairs": [list(item) for item in sorted(candidate_lengths)],
        "greedy": greedy_metrics(labels, greedy_predictions, relations),
        "likelihood": likelihood_metrics(labels, scores, relations),
        "outputs": outputs,
    }
