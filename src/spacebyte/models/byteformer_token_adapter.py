"""Expose pre-pooling token features from an official CoreNet ByteFormer."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class ByteFormerTokenAdapter(nn.Module):
    """Return normalized ByteFormer tokens and a binary attention mask.

    The wrapped model, including its official classifier, is kept intact. This
    adapter follows the official backbone path but fixes construction of the
    variable-length byte mask and stops before mean pooling and classification.
    """

    def __init__(self, byteformer: nn.Module) -> None:
        super().__init__()
        self.byteformer = byteformer
        if getattr(byteformer.classifier, "in_features", None) != 192:
            raise ValueError("Expected the official ByteFormer-Tiny 192-d classifier")
        if getattr(byteformer.classifier, "out_features", None) != 1000:
            raise ValueError("Expected the intact official 1000-class classifier")

    @staticmethod
    def _validate_bytes(byte_tensor: Tensor) -> None:
        if byte_tensor.ndim != 2:
            raise ValueError(f"Expected [B, L] bytes, got {tuple(byte_tensor.shape)}")
        if byte_tensor.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"Expected int32 or int64 bytes, got {byte_tensor.dtype}")
        if byte_tensor.shape[0] == 0 or byte_tensor.shape[1] == 0:
            raise ValueError("Byte tensor must contain at least one non-empty sample")
        if bool(((byte_tensor < -1) | (byte_tensor > 255)).any().item()):
            raise ValueError("Byte values must be in [0, 255], with -1 for padding")

        valid = byte_tensor.ne(-1)
        if bool((valid.sum(dim=1) == 0).any().item()):
            raise ValueError("Every sample must contain at least one real byte")
        # Padding must be a right-side suffix. A true value here identifies a
        # non-padding byte immediately after padding has already started.
        padding_seen = byte_tensor.eq(-1).cummax(dim=1).values
        if bool((padding_seen & valid).any().item()):
            raise ValueError("Padding value -1 must form a contiguous right suffix")

    def _get_backbone_inputs(
        self, byte_tensor: Tensor, trace: dict[str, Any]
    ) -> tuple[Tensor, Tensor]:
        self._validate_bytes(byte_tensor)
        model = self.byteformer

        # Boolean advanced indexing returns a temporary tensor in PyTorch, so
        # mask[x == -1].fill_(-inf) does not update mask. masked_fill_ does.
        padding = byte_tensor.eq(-1)
        internal_mask = torch.zeros_like(byte_tensor, dtype=torch.float32)
        internal_mask.masked_fill_(padding, float("-inf"))
        internal_mask = internal_mask.detach().requires_grad_(False)

        safe_bytes = byte_tensor.clone()
        safe_bytes.masked_fill_(padding, model.embeddings.padding_idx)
        tokens = model.embeddings(safe_bytes)
        trace["byte_input"] = list(byte_tensor.shape)
        trace["byte_embedding"] = list(tokens.shape)
        trace["byte_valid_counts"] = byte_tensor.ne(-1).sum(dim=1).tolist()

        if model.token_reduction_net is not None:
            tokens = model.token_reduction_net(tokens.permute(0, 2, 1)).permute(
                0, 2, 1
            )
            mask_windows = internal_mask.unfold(
                dimension=1,
                size=model.conv_kernel_size,
                step=model.conv_kernel_size // 2,
            )
            # A convolution output exists in an independently processed sample
            # only when its complete kernel lies inside the real byte stream.
            # CoreNet's max would retain a final partial real/padding window that
            # exists only because a longer batch peer extended the tensor.
            internal_mask = mask_windows.min(dim=-1).values
            if tokens.shape[:2] != internal_mask.shape:
                raise RuntimeError("Convolution feature and mask shapes diverged")
        trace["initial_convolution"] = list(tokens.shape)
        trace["initial_convolution_valid_counts"] = (
            torch.isfinite(internal_mask).sum(dim=1).tolist()
        )

        tokens = tokens + model.pos_embed(model.max_num_tokens)[:, : tokens.shape[1]]
        tokens = model.emb_dropout(tokens)
        return tokens, internal_mask

    def _backbone_forward(
        self, tokens: Tensor, internal_mask: Tensor, trace: dict[str, Any]
    ) -> tuple[Tensor, Tensor]:
        model = self.byteformer
        merging_shapes: list[dict[str, Any]] = []

        for layer_index, block in enumerate(model.transformer):
            # CoreNet's window block pads each local window with zero vectors.
            # Replacing variable-length batch padding by the same zero vectors
            # preserves the official no-padding path and prevents padded byte
            # embeddings from entering attention. TokenMerging also zeros these
            # positions and downsamples the mask using its official logic.
            tokens = tokens.masked_fill(
                ~torch.isfinite(internal_mask).unsqueeze(-1), 0
            )
            tokens = block(tokens, key_padding_mask=internal_mask)
            trace[f"transformer_{layer_index}"] = list(tokens.shape)

            downsampler = model.get_downsampler(layer_index)
            if downsampler is not None:
                tokens, internal_mask = downsampler(tokens, internal_mask)
                merge_record = {
                    "after_transformer": layer_index,
                    "features_shape": list(tokens.shape),
                    "valid_counts": torch.isfinite(internal_mask)
                    .sum(dim=1)
                    .tolist(),
                }
                merging_shapes.append(merge_record)
                trace[f"token_merging_after_{layer_index}"] = list(tokens.shape)

        trace["token_merging_stages"] = merging_shapes
        trace["pre_post_transformer_norm"] = list(tokens.shape)
        tokens = model.post_transformer_norm(tokens)
        trace["post_transformer_norm"] = list(tokens.shape)
        return tokens, internal_mask

    def forward_with_trace(
        self, byte_tensor: Tensor
    ) -> tuple[Tensor, Tensor, dict[str, Any]]:
        """Return token features, binary attention mask, and shape trace."""
        self._validate_bytes(byte_tensor)

        # CoreNet's shifted-window implementation at the pinned commit makes
        # its circular shift boundary depend on the padded batch length. Run
        # each true byte stream at its own length to preserve the exact
        # official no-padding computation, then pad only the final token
        # features. This is intentionally correctness-first; no parameters or
        # model modules are copied, replaced, or patched at runtime.
        byte_lengths = byte_tensor.ne(-1).sum(dim=1).tolist()
        sample_features: list[Tensor] = []
        sample_traces: list[dict[str, Any]] = []
        for sample_index, byte_length in enumerate(byte_lengths):
            sample_trace: dict[str, Any] = {"sample_index": sample_index}
            dense_bytes = byte_tensor[
                sample_index : sample_index + 1, : int(byte_length)
            ]
            tokens, internal_mask = self._get_backbone_inputs(
                dense_bytes, sample_trace
            )
            features, internal_mask = self._backbone_forward(
                tokens, internal_mask, sample_trace
            )
            if not bool(torch.isfinite(internal_mask).all().item()):
                raise RuntimeError("Dense per-sample backbone produced padding tokens")
            sample_features.append(features[0])
            sample_trace["final_valid_tokens"] = features.shape[1]
            sample_traces.append(sample_trace)

        max_tokens = max(features.shape[0] for features in sample_features)
        feature_dim = sample_features[0].shape[-1]
        features = sample_features[0].new_zeros(
            (len(sample_features), max_tokens, feature_dim)
        )
        attention_mask = torch.zeros(
            (len(sample_features), max_tokens),
            dtype=torch.long,
            device=byte_tensor.device,
        )
        for sample_index, sample in enumerate(sample_features):
            token_count = sample.shape[0]
            features[sample_index, :token_count] = sample
            attention_mask[sample_index, :token_count] = 1

        trace = {
            "batch_byte_input": list(byte_tensor.shape),
            "byte_lengths": byte_lengths,
            "per_sample": sample_traces,
            "features": list(features.shape),
            "attention_mask": list(attention_mask.shape),
            "final_valid_counts": attention_mask.sum(dim=1).tolist(),
        }
        return features, attention_mask, trace

    def forward(self, byte_tensor: Tensor) -> tuple[Tensor, Tensor]:
        features, attention_mask, _ = self.forward_with_trace(byte_tensor)
        return features, attention_mask

    def pool_and_classify(self, features: Tensor, attention_mask: Tensor) -> Tensor:
        """Reproduce ByteFormer's official masked mean and classifier path."""
        if attention_mask.shape != features.shape[:2]:
            raise ValueError(
                "Attention mask must match the first two feature dimensions"
            )
        valid = attention_mask.to(dtype=torch.bool)
        masked_features = features.masked_fill(~valid.unsqueeze(-1), 0)
        pooled = masked_features.sum(dim=1) / valid.sum(dim=1, keepdim=True)
        return self.byteformer.classifier(pooled)
