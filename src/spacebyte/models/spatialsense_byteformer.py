"""Build a SpatialSense classifier from the official ByteFormer checkpoint."""

from pathlib import Path
from typing import Any

import torch

from corenet.modeling.layers import LinearLayer
from corenet.modeling.models import get_model
from corenet.options.opts import get_training_arguments
from spacebyte.constants import RELATIONS


def build_spatialsense_byteformer(
    corenet_config: Path,
    checkpoint_path: Path,
) -> tuple[torch.nn.Module, Any]:
    opts = get_training_arguments(args=["--common.config-file", str(corenet_config)])
    setattr(opts, "model.classification.n_classes", 1000)
    model = get_model(opts=opts, category="classification", model_name="byteformer")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    input_features = state_dict["classifier.weight"].shape[1]
    model.classifier = LinearLayer(input_features, len(RELATIONS))
    torch.nn.init.trunc_normal_(model.classifier.weight, std=0.02)
    if model.classifier.bias is not None:
        torch.nn.init.zeros_(model.classifier.bias)
    setattr(opts, "model.classification.n_classes", len(RELATIONS))
    return model, opts
