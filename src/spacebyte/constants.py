"""Shared constants for the SpatialSense task."""

RELATIONS = (
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

RELATION_TO_INDEX = {name: index for index, name in enumerate(RELATIONS)}
LEFT_INDEX = RELATION_TO_INDEX["to the left of"]
RIGHT_INDEX = RELATION_TO_INDEX["to the right of"]
