"""SpatialSense preprocessing and byte-level dataset support."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

import torch
from PIL import Image, ImageDraw
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, WeightedRandomSampler

from spacebyte.constants import (LEFT_INDEX, RELATION_TO_INDEX, RELATIONS,
                                 RIGHT_INDEX)


def resolve_image_path(dataset_root: Path, url: str) -> Path:
    """Resolve an annotation URL/path to a local SpatialSense image."""
    name = Path(urlparse(url).path).name
    subdir = "nyu" if name.lower().endswith(".png") else "flickr"
    path = dataset_root / "images" / subdir / name
    if not path.is_file():
        raise FileNotFoundError(f"SpatialSense image not found: {path}")
    return path


def _pair_key(annotation: Mapping[str, Any]) -> tuple[Any, ...]:
    subject = annotation["subject"]
    obj = annotation["object"]
    return (
        subject["name"],
        tuple(subject["bbox"]),
        obj["name"],
        tuple(obj["bbox"]),
    )


def build_pair_records(
    dataset_root: Path,
    limit_per_split: int | None = None,
) -> list[dict[str, Any]]:
    """Aggregate sparse relation annotations for each ordered object pair."""
    annotations_path = dataset_root / "annotations.json"
    image_records = json.loads(annotations_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()

    for image_index, image_record in enumerate(image_records):
        split = image_record["split"]
        groups: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
        for annotation in image_record["annotations"]:
            key = _pair_key(annotation)
            if key not in groups:
                groups[key] = {
                    "subject": annotation["subject"],
                    "object": annotation["object"],
                    "labels": {},
                }
            predicate = annotation["predicate"]
            label = bool(annotation["label"])
            old_label = groups[key]["labels"].get(predicate)
            if old_label is not None and old_label != label:
                raise ValueError(
                    f"Conflicting labels for image {image_index}, pair {key}, {predicate}"
                )
            groups[key]["labels"][predicate] = label

        source_path = resolve_image_path(dataset_root, image_record["url"])
        for pair_index, group in enumerate(groups.values()):
            if limit_per_split is not None and split_counts[split] >= limit_per_split:
                continue
            target = [0.0] * len(RELATIONS)
            mask = [0.0] * len(RELATIONS)
            for predicate, label in group["labels"].items():
                relation_index = RELATION_TO_INDEX[predicate]
                target[relation_index] = float(label)
                mask[relation_index] = 1.0
            sample_id = f"{image_index:05d}_{pair_index:02d}"
            records.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "source_path": str(source_path.relative_to(dataset_root)),
                    "source_url": image_record["url"],
                    "annotation_width": image_record["width"],
                    "annotation_height": image_record["height"],
                    "subject": group["subject"],
                    "object": group["object"],
                    "target": target,
                    "mask": mask,
                }
            )
            split_counts[split] += 1
    return records


def _transform_bbox(
    bbox: Sequence[float], scale: float, offset_x: float, offset_y: float, size: int
) -> list[int]:
    y0, y1, x0, x1 = bbox
    transformed = [
        round(x0 * scale + offset_x),
        round(y0 * scale + offset_y),
        round(x1 * scale + offset_x),
        round(y1 * scale + offset_y),
    ]
    return [max(0, min(size - 1, value)) for value in transformed]


def render_pair_image(
    image: Image.Image,
    subject_bbox: Sequence[float],
    object_bbox: Sequence[float],
    output_size: int = 224,
    box_width: int = 3,
) -> Image.Image:
    """Letterbox an image and draw an ordered subject/object pair."""
    image = image.convert("RGB")
    scale = min(output_size / image.width, output_size / image.height)
    resized_width = max(1, round(image.width * scale))
    resized_height = max(1, round(image.height * scale))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    offset_x = (output_size - resized_width) // 2
    offset_y = (output_size - resized_height) // 2
    canvas = Image.new("RGB", (output_size, output_size), color=(128, 128, 128))
    canvas.paste(resized, (offset_x, offset_y))

    subject_xyxy = _transform_bbox(subject_bbox, scale, offset_x, offset_y, output_size)
    object_xyxy = _transform_bbox(object_bbox, scale, offset_x, offset_y, output_size)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(object_xyxy, outline=(0, 96, 255), width=box_width)
    draw.rectangle(subject_xyxy, outline=(255, 32, 32), width=box_width)
    return canvas


def swap_left_right_labels(
    target: Sequence[float], mask: Sequence[float]
) -> tuple[list[float], list[float]]:
    """Swap directional labels after a horizontal image flip."""
    new_target = list(target)
    new_mask = list(mask)
    new_target[LEFT_INDEX], new_target[RIGHT_INDEX] = (
        new_target[RIGHT_INDEX],
        new_target[LEFT_INDEX],
    )
    new_mask[LEFT_INDEX], new_mask[RIGHT_INDEX] = (
        new_mask[RIGHT_INDEX],
        new_mask[LEFT_INDEX],
    )
    return new_target, new_mask


def prepare_dataset(
    dataset_root: Path,
    output_root: Path,
    output_size: int = 224,
    jpeg_quality: int = 100,
    limit_per_split: int | None = None,
) -> dict[str, Any]:
    """Render pair-marked JPEGs and write sparse-label manifests."""
    records = build_pair_records(dataset_root, limit_per_split=limit_per_split)
    manifest_dir = output_root / "manifests"
    image_root = output_root / "images"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    handles = {
        split: (manifest_dir / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in ("train", "valid", "test")
    }
    split_counts: Counter[str] = Counter()
    relation_counts = {name: Counter() for name in RELATIONS}
    try:
        for record in records:
            split = record["split"]
            output_dir = image_root / split
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{record['sample_id']}.jpg"
            source_path = dataset_root / record["source_path"]
            with Image.open(source_path) as image:
                rendered = render_pair_image(
                    image,
                    record["subject"]["bbox"],
                    record["object"]["bbox"],
                    output_size=output_size,
                )
                rendered.save(output_path, format="JPEG", quality=jpeg_quality)
            record["image_path"] = str(output_path.relative_to(output_root))
            record["byte_length"] = output_path.stat().st_size
            handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
            split_counts[split] += 1
            for relation_index, relation in enumerate(RELATIONS):
                if record["mask"][relation_index]:
                    label_name = (
                        "positive" if record["target"][relation_index] else "negative"
                    )
                    relation_counts[relation][label_name] += 1
    finally:
        for handle in handles.values():
            handle.close()

    metadata = {
        "dataset": "SpatialSense",
        "relations": list(RELATIONS),
        "output_size": output_size,
        "jpeg_quality": jpeg_quality,
        "limit_per_split": limit_per_split,
        "split_pair_counts": dict(split_counts),
        "relation_counts": {
            name: dict(counts) for name, counts in relation_counts.items()
        },
        "annotations_sha256": hashlib.sha256(
            (dataset_root / "annotations.json").read_bytes()
        ).hexdigest(),
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class SpatialSenseByteDataset(Dataset[dict[str, Any]]):
    """Load pair-marked JPEGs directly as integer byte sequences."""

    def __init__(self, processed_root: Path, split: str, limit: int | None = None):
        self.processed_root = processed_root
        self.records = load_manifest(processed_root / "manifests" / f"{split}.jsonl")
        if limit is not None:
            self.records = self.records[:limit]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = self.processed_root / record["image_path"]
        byte_tensor = torch.tensor(list(image_path.read_bytes()), dtype=torch.int32)
        return {
            "samples": byte_tensor,
            "targets": torch.tensor(record["target"], dtype=torch.float32),
            "masks": torch.tensor(record["mask"], dtype=torch.float32),
            "sample_id": record["sample_id"],
        }


def collate_byte_samples(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    samples = pad_sequence(
        [item["samples"] for item in batch], batch_first=True, padding_value=-1
    )
    return {
        "samples": samples,
        "targets": torch.stack([item["targets"] for item in batch]),
        "masks": torch.stack([item["masks"] for item in batch]),
        "sample_ids": [item["sample_id"] for item in batch],
    }


def build_weighted_sampler(dataset: SpatialSenseByteDataset) -> WeightedRandomSampler:
    masks = torch.tensor([record["mask"] for record in dataset.records])
    counts = masks.sum(dim=0).clamp_min(1.0)
    relation_weights = counts.sum() / counts
    sample_weights = (masks * relation_weights).sum(dim=1) / masks.sum(dim=1)
    return WeightedRandomSampler(
        sample_weights.double(), len(dataset), replacement=True
    )
