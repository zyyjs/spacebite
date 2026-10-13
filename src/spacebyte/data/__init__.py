"""Dataset and preprocessing utilities owned by SpaceByte."""

from .spatialsense import (
    NORMALIZED_RELATIONS,
    PREPROCESS_VERSION,
    RAW_RELATIONS,
    SpatialSenseCachedFeatureDataset,
    build_tiny_overfit_manifest,
    collate_cached_features,
    encode_marked_jpeg,
    flatten_spatialsense_annotations,
    load_json,
    normalize_relation,
    parse_binary_answer,
    render_marked_image,
    resolve_image_path,
)

__all__ = [
    "NORMALIZED_RELATIONS",
    "PREPROCESS_VERSION",
    "RAW_RELATIONS",
    "SpatialSenseCachedFeatureDataset",
    "build_tiny_overfit_manifest",
    "collate_cached_features",
    "encode_marked_jpeg",
    "flatten_spatialsense_annotations",
    "load_json",
    "normalize_relation",
    "parse_binary_answer",
    "render_marked_image",
    "resolve_image_path",
]
