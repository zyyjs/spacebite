#!/usr/bin/env python3
"""Audit real SpatialSense annotations and build a deterministic tiny set."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from spacebyte.data import (
    PREPROCESS_VERSION,
    RAW_RELATIONS,
    build_tiny_overfit_manifest,
    flatten_spatialsense_annotations,
)
from spacebyte.data.spatialsense import load_json, render_record_to_jpeg_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--visualization-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261013)
    return parser.parse_args()


def nested_counts(records):
    result = {}
    for split in ("train", "valid", "test"):
        result[split] = {}
        for relation in RAW_RELATIONS:
            subset = [
                row
                for row in records
                if row["split"] == split and row["relation_raw"] == relation
            ]
            result[split][relation] = {
                "false": sum(not row["label"] for row in subset),
                "true": sum(row["label"] for row in subset),
                "total": len(subset),
            }
    return result


def main() -> None:
    args = parse_args()
    annotations_path = args.dataset_root / "annotations.json"
    image_records = load_json(annotations_path)
    records = flatten_spatialsense_annotations(args.dataset_root)
    manifest = build_tiny_overfit_manifest(records, seed=args.seed)

    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.visualization_dir.mkdir(parents=True, exist_ok=True)
    visualizations = []
    for relation in RAW_RELATIONS:
        record = next(
            row
            for row in manifest["samples"]
            if row["relation_raw"] == relation and row["label"]
        )
        jpeg, _, boxes = render_record_to_jpeg_bytes(args.dataset_root, record)
        path = args.visualization_dir / f"{relation.replace(' ', '_')}.jpg"
        path.write_bytes(bytes(jpeg.tolist()))
        visualizations.append(
            {
                "sample_id": record["sample_id"],
                "relation": relation,
                "subject_name": record["subject"].get("name"),
                "object_name": record["object"].get("name"),
                "subject_red_xyxy": boxes["subject_xyxy"],
                "object_blue_xyxy": boxes["object_xyxy"],
                "path": str(path.resolve()),
                "jpeg_byte_length": jpeg.numel(),
            }
        )

    image_split_counts = Counter(row["split"] for row in image_records)
    annotation_split_counts = Counter(row["split"] for row in records)
    labels = {
        split: {
            "false": sum(row["split"] == split and not row["label"] for row in records),
            "true": sum(row["split"] == split and row["label"] for row in records),
        }
        for split in ("train", "valid", "test")
    }
    annotations_per_image = Counter(len(row["annotations"]) for row in image_records)
    pair_counts = Counter(
        (
            row["image_path"],
            tuple(row["subject"]["bbox"]),
            tuple(row["object"]["bbox"]),
        )
        for row in records
    )
    image_paths_by_index = {}
    for record in records:
        image_paths_by_index.setdefault(
            record["image_record_index"], record["image_path"]
        )
    dimension_mismatches = []
    annotated_images_checked = 0
    for image_index, image_record in enumerate(image_records):
        if not image_record["annotations"]:
            continue
        annotated_images_checked += 1
        source_path = args.dataset_root / image_paths_by_index[image_index]
        with Image.open(source_path) as image:
            actual_size = image.size
        declared_size = (image_record["width"], image_record["height"])
        if actual_size != declared_size:
            dimension_mismatches.append(
                {
                    "image_record_index": image_index,
                    "declared_size": declared_size,
                    "actual_size": actual_size,
                }
            )
    report = {
        "dataset": "SpatialSense",
        "task_semantics": "candidate_relation_plus_explicit_boolean_label",
        "generation_target": "yes/no",
        "unannotated_relations_are_not_negative": True,
        "annotations_path": str(annotations_path.resolve()),
        "annotations_sha256": hashlib.sha256(annotations_path.read_bytes()).hexdigest(),
        "image_directories": [
            str((args.dataset_root / "images" / "flickr").resolve()),
            str((args.dataset_root / "images" / "nyu").resolve()),
        ],
        "top_level_type": type(image_records).__name__,
        "image_record_fields": sorted(image_records[0]),
        "annotation_fields": sorted(image_records[0]["annotations"][0]),
        "entity_fields": sorted(image_records[0]["annotations"][0]["subject"]),
        "image_record_count": len(image_records),
        "image_record_counts_by_split": dict(image_split_counts),
        "annotation_count": len(records),
        "annotation_counts_by_split": dict(annotation_split_counts),
        "explicit_label_counts_by_split": labels,
        "relation_counts": nested_counts(records),
        "unique_annotation_ids": len({row["sample_id"] for row in records}),
        "images_with_multiple_annotations": sum(
            count for size, count in annotations_per_image.items() if size > 1
        ),
        "max_annotations_per_image": max(annotations_per_image),
        "ordered_object_pair_count": len(pair_counts),
        "pairs_with_multiple_candidate_relations": sum(
            count > 1 for count in pair_counts.values()
        ),
        "max_candidate_relations_per_pair": max(pair_counts.values()),
        "bbox": {
            "annotation_order": "[y_min, y_max, x_min, x_max]",
            "annotated_images_checked": annotated_images_checked,
            "annotation_vs_decoded_dimension_mismatches": len(
                dimension_mismatches
            ),
            "annotation_to_actual_image_scaling": (
                "implemented defensively; ratio is 1.0 for every annotated "
                "image in this local release"
            ),
            "resize": "aspect-preserving letterbox to 224x224 with bicubic interpolation",
            "letterbox_rgb": [128, 128, 128],
            "subject_color_rgb": [255, 32, 32],
            "object_color_rgb": [0, 96, 255],
            "line_width_pixels": 3,
            "out_of_bounds": "clamp each transformed coordinate to [0,223]",
            "draw_order": "blue object first, red subject second",
        },
        "source_images_have_task_boxes": False,
        "source_box_evidence": (
            "Boxes are structured annotation fields; the archived verified "
            "pipeline explicitly rendered them onto raw source images."
        ),
        "jpeg": {
            "encoding": "PIL JPEG",
            "quality": 100,
            "optimization": False,
            "preprocess_version": PREPROCESS_VERSION,
        },
        "tiny_manifest": {
            "path": str(args.manifest_output.resolve()),
            "seed": manifest["seed"],
            "sample_count": manifest["sample_count"],
            "class_counts": manifest["class_counts"],
            "unique_sample_ids": len(
                {row["sample_id"] for row in manifest["samples"]}
            ),
        },
        "visualizations": visualizations,
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
