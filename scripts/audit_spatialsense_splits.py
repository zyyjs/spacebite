#!/usr/bin/env python3
"""Audit every requested SpatialSense train/valid/test leakage mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

from spacebyte.data import flatten_spatialsense_annotations, sha256_file
from spacebyte.data.spatialsense import render_record_to_jpeg_bytes


SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def exact_key(record) -> str:
    return canonical(
        [
            record["image_path"],
            record["subject"]["bbox"],
            record["object"]["bbox"],
            record["relation_raw"],
            record["label"],
        ]
    )


def pair_key(record) -> str:
    return canonical(
        [
            record["image_path"],
            record["subject"]["bbox"],
            record["object"]["bbox"],
        ]
    )


def pairwise_overlap(
    values: dict[str, dict[str, list[dict[str, Any]]]], example_limit: int = 5
) -> dict[str, Any]:
    report = {}
    for left, right in combinations(SPLITS, 2):
        shared = sorted(set(values[left]) & set(values[right]))
        report[f"{left}__{right}"] = {
            "unique_overlap_count": len(shared),
            "examples": [
                {
                    "key": key,
                    left: values[left][key][:2],
                    right: values[right][key][:2],
                }
                for key in shared[:example_limit]
            ],
        }
    return report


def indexed(
    records: list[dict[str, Any]], key_function: Callable[[dict[str, Any]], str]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result = {split: defaultdict(list) for split in SPLITS}
    for record in records:
        result[record["split"]][key_function(record)].append(
            {
                "sample_id": record["sample_id"],
                "image_path": record["image_path"],
                "relation": record["relation_raw"],
                "label": record["label"],
            }
        )
    return result


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    records = flatten_spatialsense_annotations(args.dataset_root)
    split_counts = Counter(record["split"] for record in records)

    sample_ids = indexed(records, lambda row: row["sample_id"])
    exact_annotations = indexed(records, exact_key)
    raw_paths = indexed(records, lambda row: row["image_path"])

    unique_paths = sorted({record["image_path"] for record in records})
    raw_sha_by_path = {
        path: sha256_file(args.dataset_root / path) for path in unique_paths
    }
    raw_hashes = indexed(records, lambda row: raw_sha_by_path[row["image_path"]])

    processed_sha_by_pair = {}
    processed_hashes = {split: defaultdict(list) for split in SPLITS}
    for index, record in enumerate(records, start=1):
        visual_pair = pair_key(record)
        if visual_pair not in processed_sha_by_pair:
            jpeg, _, _ = render_record_to_jpeg_bytes(args.dataset_root, record)
            processed_sha_by_pair[visual_pair] = hashlib.sha256(
                bytes(jpeg.tolist())
            ).hexdigest()
        processed_hashes[record["split"]][
            processed_sha_by_pair[visual_pair]
        ].append(
            {
                "sample_id": record["sample_id"],
                "image_path": record["image_path"],
                "relation": record["relation_raw"],
                "label": record["label"],
            }
        )
        if index % 2000 == 0:
            print(f"processed {index}/{len(records)} annotations", flush=True)

    within_split_duplicates = {}
    for split in SPLITS:
        duplicate_keys = {
            key: rows for key, rows in exact_annotations[split].items() if len(rows) > 1
        }
        within_split_duplicates[split] = {
            "duplicate_tuple_count": len(duplicate_keys),
            "extra_annotation_count": sum(len(rows) - 1 for rows in duplicate_keys.values()),
            "examples": list(duplicate_keys.items())[:5],
        }

    relation_groups = defaultdict(list)
    for record in records:
        relation_groups[(record["split"], pair_key(record))].append(record)
    differing_relation_groups = []
    for (split, key), rows in relation_groups.items():
        relations = sorted({row["relation_raw"] for row in rows})
        if len(relations) > 1:
            differing_relation_groups.append(
                {
                    "split": split,
                    "pair_key": key,
                    "relations": relations,
                    "sample_ids": [row["sample_id"] for row in rows],
                }
            )

    overlap = {
        "sample_id": pairwise_overlap(sample_ids),
        "exact_annotation_tuple": pairwise_overlap(exact_annotations),
        "raw_image_path": pairwise_overlap(raw_paths),
        "raw_image_sha256": pairwise_overlap(raw_hashes),
        "processed_jpeg_sha256": pairwise_overlap(processed_hashes),
    }
    exact_cross_split_count = sum(
        row["unique_overlap_count"]
        for row in overlap["exact_annotation_tuple"].values()
    )
    report = {
        "dataset_root": str(args.dataset_root.resolve()),
        "sample_counts": dict(split_counts),
        "unique_raw_image_paths": len(unique_paths),
        "unique_raw_image_sha256": len(set(raw_sha_by_path.values())),
        "unique_processed_visual_pairs": len(processed_sha_by_pair),
        "unique_processed_jpeg_sha256": len(set(processed_sha_by_pair.values())),
        "overlap": overlap,
        "within_split_exact_annotation_duplicates": within_split_duplicates,
        "same_subject_object_with_different_candidate_relations": {
            "group_count": len(differing_relation_groups),
            "sample_count": sum(
                len(row["sample_ids"]) for row in differing_relation_groups
            ),
            "counts_by_split": dict(
                Counter(row["split"] for row in differing_relation_groups)
            ),
            "examples": differing_relation_groups[:10],
        },
        "exact_cross_split_annotation_overlap_count": exact_cross_split_count,
        "safe_to_continue": exact_cross_split_count == 0,
        "duration_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if exact_cross_split_count:
        raise RuntimeError(
            "Official splits contain exact annotation tuple overlap; stop before cache"
        )


if __name__ == "__main__":
    main()
