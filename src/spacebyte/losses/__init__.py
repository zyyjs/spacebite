"""Loss functions for sparse SpatialSense supervision."""

from .masked_bce import masked_binary_cross_entropy

__all__ = ["masked_binary_cross_entropy"]
