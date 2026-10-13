"""Auditable SpatialSense candidate-relation instruction pipeline.

SpatialSense annotations are binary judgements of an explicitly supplied
candidate predicate. They are deliberately kept as independent yes/no samples;
unannotated predicates are never synthesized as negative labels.
"""

from __future__ import annotations

import io
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import Dataset


RAW_RELATIONS = (
    "above",
    "behind",
    "in",
    "in front of",
    "next to",
    "on",
    "to the left of",
    "to the right of",
    "under",
)
RELATION_NORMALIZATION = {
    "above": "above",
    "behind": "behind",
    "in": "in",
    "in front of": "in front of",
    "next to": "next to",
    "on": "on",
    "to the left of": "left of",
    "to the right of": "right of",
    "under": "under",
}
NORMALIZED_RELATIONS = tuple(RELATION_NORMALIZATION[item] for item in RAW_RELATIONS)
PREPROCESS_VERSION = "spatialsense-red-subject-blue-object-letterbox224-jpegq100-v1"
TINY_MANIFEST_VERSION = 1
DEFAULT_TINY_SEED = 20261013
DEFAULT_PER_RELATION_PER_LABEL = 2
SUBJECT_COLOR = (255, 32, 32)
OBJECT_COLOR = (0, 96, 255)
LETTERBOX_COLOR = (128, 128, 128)
QUESTION_TEMPLATE = (
    "Is the object in the red box {relation} the object in the blue box? "
    "Answer yes or no."
)
VISUAL_PREPROCESS_CONFIG = {
    "output_size": 224,
    "resize": "aspect_preserving_bicubic_letterbox",
    "letterbox_rgb": list(LETTERBOX_COLOR),
    "subject_rgb": list(SUBJECT_COLOR),
    "object_rgb": list(OBJECT_COLOR),
    "box_width": 3,
    "bbox_order": "ymin_ymax_xmin_xmax",
    "jpeg_quality": 100,
    "jpeg_encoder": "PIL_unoptimized",
    "preprocessing_version": PREPROCESS_VERSION,
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def visual_cache_identity(
    record: Mapping[str, Any],
    *,
    raw_image_sha256: str,
    byteformer_checkpoint_sha256: str,
) -> tuple[str, dict[str, Any]]:
    """Build the complete visual identity required for safe feature reuse."""

    payload = {
        "raw_image_sha256": raw_image_sha256,
        "subject_bbox": [float(value) for value in record["subject"]["bbox"]],
        "object_bbox": [float(value) for value in record["object"]["bbox"]],
        "preprocessing": VISUAL_PREPROCESS_CONFIG,
        "byteformer_checkpoint_sha256": byteformer_checkpoint_sha256,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), payload


def atomic_torch_save(value: Any, path: Path) -> None:
    """Write a torch artifact atomically in the destination directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_feature_cache_file(
    path: Path, *, expected_visual_cache_key: str | None = None
) -> tuple[bool, str]:
    """Detect absent, truncated, mismatched, or structurally invalid cache files."""

    if not path.is_file():
        return False, "missing"
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        return False, f"unreadable:{type(error).__name__}"
    required = {
        "visual_cache_key",
        "byte_features",
        "byte_attention_mask",
        "valid_token_count",
        "jpeg_byte_length",
        "processed_jpeg_sha256",
        "preprocessing_version",
    }
    if not isinstance(value, dict) or not required.issubset(value):
        return False, "missing_fields"
    if (
        expected_visual_cache_key is not None
        and value["visual_cache_key"] != expected_visual_cache_key
    ):
        return False, "cache_key_mismatch"
    features = value["byte_features"]
    mask = value["byte_attention_mask"]
    if (
        not torch.is_tensor(features)
        or features.ndim != 2
        or features.shape[1] != 192
        or features.dtype != torch.float16
    ):
        return False, "invalid_features"
    if not torch.is_tensor(mask) or mask.shape != features.shape[:1]:
        return False, "invalid_mask_shape"
    if not bool(((mask == 0) | (mask == 1)).all().item()):
        return False, "invalid_mask_values"
    if not bool(torch.isfinite(features).all().item()):
        return False, "nonfinite_features"
    if int(mask.sum().item()) != int(value["valid_token_count"]):
        return False, "valid_token_count_mismatch"
    return True, "valid"


def normalize_relation(predicate: str) -> str:
    try:
        return RELATION_NORMALIZATION[predicate]
    except KeyError as error:
        raise ValueError(f"Unknown SpatialSense predicate: {predicate!r}") from error


def resolve_image_path(dataset_root: Path, url: str) -> Path:
    name = Path(urlparse(url).path).name
    subdirectory = "nyu" if name.lower().endswith(".png") else "flickr"
    path = dataset_root / "images" / subdirectory / name
    if not path.is_file() or path.name.startswith("._"):
        raise FileNotFoundError(f"SpatialSense image not found: {path}")
    return path


def _validated_bbox(value: Any, *, sample_id: str, role: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{sample_id} {role} bbox must be [y0,y1,x0,x1]")
    result = [float(item) for item in value]
    y0, y1, x0, x1 = result
    if not all(np.isfinite(item) for item in result) or y0 > y1 or x0 > x1:
        raise ValueError(f"{sample_id} has invalid {role} bbox: {value}")
    return result


def flatten_spatialsense_annotations(dataset_root: Path) -> list[dict[str, Any]]:
    """Return one record per annotated candidate relation."""

    annotation_path = dataset_root / "annotations.json"
    image_records = load_json(annotation_path)
    if not isinstance(image_records, list):
        raise TypeError("SpatialSense annotations.json must contain a list")
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for image_index, image_record in enumerate(image_records):
        required_image = {"url", "height", "width", "annotations", "split"}
        if not required_image.issubset(image_record):
            raise ValueError(f"Image record {image_index} is missing required fields")
        split = image_record["split"]
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Unexpected SpatialSense split: {split!r}")
        relative_image = resolve_image_path(dataset_root, image_record["url"]).relative_to(
            dataset_root
        )
        for annotation_index, annotation in enumerate(image_record["annotations"]):
            required = {"_id", "predicate", "subject", "object", "label"}
            if not required.issubset(annotation):
                raise ValueError(
                    f"Annotation {image_index}:{annotation_index} is missing fields"
                )
            sample_id = str(annotation["_id"])
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate SpatialSense annotation id: {sample_id}")
            seen_ids.add(sample_id)
            if type(annotation["label"]) is not bool:
                raise TypeError(f"{sample_id} label must be an explicit boolean")
            raw_relation = str(annotation["predicate"])
            relation = normalize_relation(raw_relation)
            subject = dict(annotation["subject"])
            obj = dict(annotation["object"])
            subject["bbox"] = _validated_bbox(
                subject.get("bbox"), sample_id=sample_id, role="subject"
            )
            obj["bbox"] = _validated_bbox(
                obj.get("bbox"), sample_id=sample_id, role="object"
            )
            answer = "yes" if annotation["label"] else "no"
            records.append(
                {
                    "sample_id": sample_id,
                    "image_record_index": image_index,
                    "annotation_index": annotation_index,
                    "split": split,
                    "image_path": str(relative_image),
                    "source_url": image_record["url"],
                    "annotation_width": int(image_record["width"]),
                    "annotation_height": int(image_record["height"]),
                    "subject": subject,
                    "object": obj,
                    "relation_raw": raw_relation,
                    "relation": relation,
                    "label": annotation["label"],
                    "answer": answer,
                    "question": QUESTION_TEMPLATE.format(relation=relation),
                }
            )
    return records


def _bbox_to_letterboxed_xyxy(
    bbox: Sequence[float],
    *,
    annotation_size: tuple[int, int],
    actual_size: tuple[int, int],
    resize_scale: float,
    offset: tuple[int, int],
    output_size: int,
) -> list[int]:
    """Convert annotation [y0,y1,x0,x1] to clamped output xyxy."""

    y0, y1, x0, x1 = bbox
    annotation_width, annotation_height = annotation_size
    actual_width, actual_height = actual_size
    if annotation_width <= 0 or annotation_height <= 0:
        raise ValueError("Annotation width and height must be positive")
    x_ratio = actual_width / annotation_width
    y_ratio = actual_height / annotation_height
    offset_x, offset_y = offset
    values = [
        round(x0 * x_ratio * resize_scale + offset_x),
        round(y0 * y_ratio * resize_scale + offset_y),
        round(x1 * x_ratio * resize_scale + offset_x),
        round(y1 * y_ratio * resize_scale + offset_y),
    ]
    return [max(0, min(output_size - 1, item)) for item in values]


def render_marked_image(
    image: Image.Image,
    *,
    subject_bbox: Sequence[float],
    object_bbox: Sequence[float],
    annotation_width: int,
    annotation_height: int,
    output_size: int = 224,
    box_width: int = 3,
) -> tuple[Image.Image, dict[str, list[int]]]:
    """Letterbox RGB input and draw red subject plus blue object boxes."""

    if output_size <= 0 or box_width <= 0:
        raise ValueError("output_size and box_width must be positive")
    image = image.convert("RGB")
    actual_size = image.size
    scale = min(output_size / image.width, output_size / image.height)
    resized_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    resized = image.resize(resized_size, Image.Resampling.BICUBIC)
    offset = (
        (output_size - resized_size[0]) // 2,
        (output_size - resized_size[1]) // 2,
    )
    canvas = Image.new("RGB", (output_size, output_size), LETTERBOX_COLOR)
    canvas.paste(resized, offset)
    common = {
        "annotation_size": (annotation_width, annotation_height),
        "actual_size": actual_size,
        "resize_scale": scale,
        "offset": offset,
        "output_size": output_size,
    }
    subject_xyxy = _bbox_to_letterboxed_xyxy(subject_bbox, **common)
    object_xyxy = _bbox_to_letterboxed_xyxy(object_bbox, **common)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(object_xyxy, outline=OBJECT_COLOR, width=box_width)
    draw.rectangle(subject_xyxy, outline=SUBJECT_COLOR, width=box_width)
    return canvas, {"subject_xyxy": subject_xyxy, "object_xyxy": object_xyxy}


def encode_marked_jpeg(image: Image.Image, quality: int = 100) -> Tensor:
    """Encode a rendered RGB image exactly as an unoptimized PIL JPEG stream."""

    if quality != 100:
        raise ValueError("Official JPEG-Q100 ByteFormer requires quality=100")
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    values = np.frombuffer(buffer.getvalue(), dtype=np.uint8).copy()
    return torch.from_numpy(values).to(dtype=torch.int32)


def render_record_to_jpeg_bytes(
    dataset_root: Path, record: Mapping[str, Any]
) -> tuple[Tensor, Image.Image, dict[str, list[int]]]:
    with Image.open(dataset_root / record["image_path"]) as source:
        rendered, boxes = render_marked_image(
            source,
            subject_bbox=record["subject"]["bbox"],
            object_bbox=record["object"]["bbox"],
            annotation_width=record["annotation_width"],
            annotation_height=record["annotation_height"],
        )
    return encode_marked_jpeg(rendered), rendered, boxes


def build_tiny_overfit_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_TINY_SEED,
    per_relation_per_label: int = DEFAULT_PER_RELATION_PER_LABEL,
) -> dict[str, Any]:
    """Choose a fixed train-only set balanced by predicate and boolean label."""

    grouped: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["split"] == "train":
            grouped[(record["relation_raw"], bool(record["label"]))].append(
                dict(record)
            )
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for relation in RAW_RELATIONS:
        for label in (False, True):
            candidates = sorted(grouped[(relation, label)], key=lambda row: row["sample_id"])
            rng.shuffle(candidates)
            if len(candidates) < per_relation_per_label:
                raise RuntimeError(
                    f"Not enough train examples for {relation!r}, label={label}"
                )
            selected.extend(candidates[:per_relation_per_label])
    selected.sort(key=lambda row: (RAW_RELATIONS.index(row["relation_raw"]), row["label"], row["sample_id"]))
    ids = [row["sample_id"] for row in selected]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Tiny overfit selection contains duplicate sample ids")
    counts = Counter((row["relation_raw"], row["answer"]) for row in selected)
    return {
        "manifest_version": TINY_MANIFEST_VERSION,
        "task": "candidate_relation_binary_generation",
        "source_split": "train",
        "seed": seed,
        "per_relation_per_label": per_relation_per_label,
        "sample_count": len(selected),
        "preprocess_version": PREPROCESS_VERSION,
        "class_counts": {
            relation: {
                "no": counts[(relation, "no")],
                "yes": counts[(relation, "yes")],
            }
            for relation in RAW_RELATIONS
        },
        "samples": selected,
    }


def parse_binary_answer(text: str) -> str | None:
    match = re.search(r"\b(yes|no)\b", text.strip().lower())
    return match.group(1) if match else None


class SpatialSenseCachedFeatureDataset(Dataset[dict[str, Any]]):
    """Load the fixed manifest and independently stored ByteFormer features."""

    def __init__(self, manifest_path: Path, cache_root: Path) -> None:
        manifest = load_json(manifest_path)
        self.records = list(manifest["samples"])
        self.cache_root = cache_root

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        cached = torch.load(
            self.cache_root / f"{record['sample_id']}.pt",
            map_location="cpu",
            weights_only=True,
        )
        if cached["sample_id"] != record["sample_id"]:
            raise RuntimeError("Cached feature/sample manifest mismatch")
        return {
            "sample_id": record["sample_id"],
            "byte_features": cached["byte_features"],
            "byte_attention_mask": cached["byte_attention_mask"].long(),
            "question": record["question"],
            "answer": record["answer"],
            "relation": record["relation"],
            "relation_raw": record["relation_raw"],
            "label": bool(record["label"]),
            "split": record.get("split", "train"),
        }


class SpatialSenseFullCachedDataset(Dataset[dict[str, Any]]):
    """Read one official split from the full annotation-to-visual-cache manifest.

    The manifest deliberately has one row per annotation even when multiple
    annotations reuse a deduplicated visual feature file.  ``expected_split``
    is mandatory so a caller cannot silently mix train, validation, and test.
    """

    def __init__(
        self,
        manifest_path: Path,
        cache_root: Path,
        *,
        expected_split: str,
    ) -> None:
        if expected_split not in {"train", "valid", "test"}:
            raise ValueError(f"Unsupported split: {expected_split!r}")
        rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        wrong = [row["sample_id"] for row in rows if row["split"] != expected_split]
        if wrong:
            raise RuntimeError(
                f"{manifest_path} contains {len(wrong)} rows outside "
                f"the expected {expected_split!r} split"
            )
        sample_ids = [row["sample_id"] for row in rows]
        if len(sample_ids) != len(set(sample_ids)):
            raise RuntimeError(f"Duplicate sample IDs in {manifest_path}")
        self.records = rows
        self.cache_root = cache_root
        self.expected_split = expected_split

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = self.cache_root / record["cache_relative_path"]
        cached = torch.load(path, map_location="cpu", weights_only=True)
        if cached["visual_cache_key"] != record["visual_cache_key"]:
            raise RuntimeError(
                f"Visual cache key mismatch for sample {record['sample_id']}"
            )
        return {
            "sample_id": record["sample_id"],
            "split": record["split"],
            "visual_cache_key": record["visual_cache_key"],
            "byte_features": cached["byte_features"],
            "byte_attention_mask": cached["byte_attention_mask"].long(),
            "question": record["question"],
            "answer": record["answer"],
            "relation": record["relation"],
            "relation_raw": record["relation_raw"],
            "label": bool(record["label"]),
        }


def collate_cached_features(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty feature batch")
    max_tokens = max(item["byte_features"].shape[0] for item in batch)
    feature_dim = batch[0]["byte_features"].shape[1]
    features = torch.zeros(
        (len(batch), max_tokens, feature_dim), dtype=batch[0]["byte_features"].dtype
    )
    mask = torch.zeros((len(batch), max_tokens), dtype=torch.long)
    for index, item in enumerate(batch):
        length = item["byte_features"].shape[0]
        features[index, :length] = item["byte_features"]
        mask[index, :length] = item["byte_attention_mask"]
    return {
        "sample_ids": [item["sample_id"] for item in batch],
        "byte_features": features,
        "byte_attention_mask": mask,
        "questions": [item["question"] for item in batch],
        "answers": [item["answer"] for item in batch],
        "relations": [item["relation"] for item in batch],
        "relations_raw": [item.get("relation_raw", item["relation"]) for item in batch],
        "labels": [bool(item.get("label", item["answer"] == "yes")) for item in batch],
        "splits": [item.get("split") for item in batch],
        "visual_cache_keys": [item.get("visual_cache_key") for item in batch],
    }
