"""Cached EVA features, Pixel-only evaluation, and auditable visual mappings."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .answer_only import binary_answer_token_ids
from .data import BLIP_IMAGE_MEAN, BLIP_IMAGE_STD, VISUAL_PREPROCESS_CONFIG
from .evaluation import likelihood_metrics
from .models import PixelVicuna


PIXEL_CACHE_VERSION = 1


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pixel_visual_cache_identity(
    record: Mapping[str, Any], *, raw_image_sha256: str, eva_checkpoint_sha256: str
) -> tuple[str, dict[str, Any]]:
    payload = {
        "cache_version": PIXEL_CACHE_VERSION,
        "raw_image_sha256": raw_image_sha256,
        "subject_bbox": [float(value) for value in record["subject"]["bbox"]],
        "object_bbox": [float(value) for value in record["object"]["bbox"]],
        "annotation_width": int(record["annotation_width"]),
        "annotation_height": int(record["annotation_height"]),
        "marked_jpeg_preprocessing": VISUAL_PREPROCESS_CONFIG,
        "pixel_normalization": {
            "mean": list(BLIP_IMAGE_MEAN),
            "std": list(BLIP_IMAGE_STD),
        },
        "eva_checkpoint_sha256": eva_checkpoint_sha256,
        "feature_location": "EVA-CLIP-G final token sequence before InstructBLIP ln_vision",
    }
    return canonical_sha256(payload), payload


class SpatialSensePixelCachedDataset(Dataset[dict[str, Any]]):
    """One split of sample metadata mapped to deduplicated frozen-EVA tensors."""

    def __init__(self, manifest_path: Path, cache_root: Path, *, expected_split: str):
        if expected_split not in {"train", "valid"}:
            raise ValueError("Pixel training only accepts train or valid")
        rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows or any(row["split"] != expected_split for row in rows):
            raise RuntimeError(f"Pixel manifest is empty or mixes splits: {manifest_path}")
        identifiers = [row["sample_id"] for row in rows]
        if len(identifiers) != len(set(identifiers)):
            raise RuntimeError("Pixel cache manifest has duplicate sample IDs")
        self.records = rows
        self.cache_root = cache_root
        self.expected_split = expected_split

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        artifact = torch.load(
            self.cache_root / row["feature_relative_path"],
            map_location="cpu",
            weights_only=True,
        )
        if artifact["pixel_cache_key"] != row["pixel_cache_key"]:
            raise RuntimeError(f"Pixel cache key mismatch for {row['sample_id']}")
        features = artifact["vision_features"]
        if features.dtype != torch.float16 or features.ndim != 2:
            raise RuntimeError(f"Invalid cached EVA feature for {row['sample_id']}")
        return {
            "sample_id": row["sample_id"],
            "source_sample_id": row["sample_id"],
            "split": row["split"],
            "pixel_cache_key": row["pixel_cache_key"],
            "source_pixel_cache_key": row["pixel_cache_key"],
            "vision_features": features,
            "question": row["question"],
            "answer": row["answer"],
            "relation": row["relation"],
            "relation_raw": row["relation_raw"],
            "label": bool(row["label"]),
        }


class PermutedPixelCachedDataset(Dataset[dict[str, Any]]):
    """Keep text/labels while selecting visual features via a global mapping."""

    def __init__(self, dataset: SpatialSensePixelCachedDataset, mapping: Sequence[int]):
        if sorted(mapping) != list(range(len(dataset))):
            raise ValueError("Shuffled-Pixel mapping is not a full permutation")
        if any(index == source for index, source in enumerate(mapping)):
            raise ValueError("Shuffled-Pixel mapping contains a sample self mapping")
        keys = [row["pixel_cache_key"] for row in dataset.records]
        if any(keys[index] == keys[source] for index, source in enumerate(mapping)):
            raise ValueError("Shuffled-Pixel mapping contains a visual self mapping")
        self.dataset = dataset
        self.mapping = list(mapping)
        self.records = dataset.records

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        target = self.dataset[index]
        source_index = self.mapping[index]
        source = self.dataset[source_index]
        target["vision_features"] = source["vision_features"]
        target["source_sample_id"] = source["sample_id"]
        target["source_pixel_cache_key"] = source["pixel_cache_key"]
        return target


def collate_pixel_features(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty Pixel batch")
    shapes = {tuple(item["vision_features"].shape) for item in batch}
    if len(shapes) != 1:
        raise RuntimeError(f"Cached EVA shapes differ within a batch: {sorted(shapes)}")
    features = torch.stack([item["vision_features"] for item in batch])
    return {
        "sample_ids": [item["sample_id"] for item in batch],
        "source_sample_ids": [item["source_sample_id"] for item in batch],
        "pixel_cache_keys": [item["pixel_cache_key"] for item in batch],
        "source_pixel_cache_keys": [item["source_pixel_cache_key"] for item in batch],
        "vision_features": features,
        "vision_attention_mask": torch.ones(features.shape[:2], dtype=torch.long),
        "questions": [item["question"] for item in batch],
        "answers": [item["answer"] for item in batch],
        "relations_raw": [item["relation_raw"] for item in batch],
        "labels": [bool(item["label"]) for item in batch],
    }


@torch.no_grad()
def evaluate_pixel_answer_only(
    model: PixelVicuna,
    dataset: Dataset,
    *,
    device: torch.device,
    batch_size: int,
    pixel_condition: str = "normal",
    num_workers: int = 0,
) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_pixel_features,
        pin_memory=num_workers > 0,
    )
    token_ids = binary_answer_token_ids(model.llm_tokenizer)
    labels: list[bool] = []
    relations: list[str] = []
    scores: list[float] = []
    outputs: list[dict[str, Any]] = []
    started = time.perf_counter()
    for batch in loader:
        features = batch["vision_features"].to(device, non_blocking=True)
        mask = batch["vision_attention_mask"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prepared = model.prepare_inputs_from_vision_features(
                features,
                mask,
                batch["questions"],
                answers=None,
                pixel_condition=pixel_condition,
            )
            result = model.vicuna(
                inputs_embeds=prepared.inputs_embeds,
                attention_mask=prepared.attention_mask,
                use_cache=False,
                return_dict=True,
            )
        positions = prepared.attention_mask.sum(1).long() - 1
        next_logits = result.logits[
            torch.arange(result.logits.shape[0], device=device), positions
        ]
        batch_scores = next_logits[:, token_ids["yes"]] - next_logits[:, token_ids["no"]]
        batch_scores_cpu = batch_scores.float().cpu().tolist()
        labels.extend(batch["labels"])
        relations.extend(batch["relations_raw"])
        scores.extend(batch_scores_cpu)
        for index, score in enumerate(batch_scores_cpu):
            outputs.append(
                {
                    "sample_id": batch["sample_ids"][index],
                    "source_sample_id": batch["source_sample_ids"][index],
                    "pixel_cache_key": batch["pixel_cache_keys"][index],
                    "source_pixel_cache_key": batch["source_pixel_cache_keys"][index],
                    "relation": batch["relations_raw"][index],
                    "label": batch["labels"][index],
                    "score_logit_yes_minus_no": score,
                    "prediction": "yes" if score > 0 else "no",
                }
            )
    return {
        "sample_count": len(labels),
        "pixel_condition": pixel_condition,
        "duration_seconds": time.perf_counter() - started,
        "logit_yes_minus_no": likelihood_metrics(labels, scores, relations),
        "outputs": outputs,
    }
