import json
from pathlib import Path

from PIL import Image

from spacebyte.constants import LEFT_INDEX, RELATION_TO_INDEX, RIGHT_INDEX
from spacebyte.data.spatialsense import (SpatialSenseByteDataset,
                                         build_pair_records,
                                         collate_byte_samples, prepare_dataset,
                                         render_pair_image,
                                         swap_left_right_labels)


def make_tiny_dataset(root: Path) -> None:
    image_dir = root / "images" / "flickr"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (100, 50), color=(240, 240, 240)).save(image_dir / "sample.jpg")
    annotation = {
        "url": "https://example.test/sample.jpg",
        "nsid": "test",
        "height": 50,
        "width": 100,
        "split": "train",
        "annotations": [
            {
                "_id": "1",
                "predicate": "to the left of",
                "subject": {"name": "cat", "bbox": [5, 30, 5, 30], "x": 10, "y": 10},
                "object": {"name": "box", "bbox": [5, 30, 60, 90], "x": 70, "y": 10},
                "label": True,
            },
            {
                "_id": "2",
                "predicate": "next to",
                "subject": {"name": "cat", "bbox": [5, 30, 5, 30], "x": 10, "y": 10},
                "object": {"name": "box", "bbox": [5, 30, 60, 90], "x": 70, "y": 10},
                "label": False,
            },
        ],
    }
    (root / "annotations.json").write_text(json.dumps([annotation]), encoding="utf-8")


def test_grouping_and_sparse_targets(tmp_path):
    make_tiny_dataset(tmp_path)
    records = build_pair_records(tmp_path)
    assert len(records) == 1
    record = records[0]
    assert sum(record["mask"]) == 2
    assert record["target"][LEFT_INDEX] == 1
    assert record["target"][RELATION_TO_INDEX["next to"]] == 0


def test_prepare_and_collate(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "processed"
    make_tiny_dataset(source)
    metadata = prepare_dataset(source, output)
    assert metadata["split_pair_counts"] == {"train": 1}
    dataset = SpatialSenseByteDataset(output, "train")
    batch = collate_byte_samples([dataset[0], dataset[0]])
    assert batch["samples"].shape[0] == 2
    assert batch["targets"].shape == (2, 9)
    assert batch["masks"].sum().item() == 4


def test_render_and_left_right_swap():
    image = Image.new("RGB", (100, 50), color="white")
    rendered = render_pair_image(image, [5, 30, 5, 30], [5, 30, 60, 90])
    assert rendered.size == (224, 224)
    target = [0.0] * 9
    mask = [0.0] * 9
    target[LEFT_INDEX] = 1.0
    mask[LEFT_INDEX] = 1.0
    flipped_target, flipped_mask = swap_left_right_labels(target, mask)
    assert flipped_target[RIGHT_INDEX] == 1.0
    assert flipped_mask[RIGHT_INDEX] == 1.0
    assert flipped_mask[LEFT_INDEX] == 0.0
