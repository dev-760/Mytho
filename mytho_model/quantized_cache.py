"""
Hierarchical Quantized KV Cache — inspired by QuantSpec (Li et al., 2025).

Implements a two-tier quantisation scheme for the MLA latent cache:
  • **Hot tier**: recent / high-attention tokens stored at full precision.
  • **Cold tier**: older tokens quantised to INT8 with per-channel
    scale + zero-point, reducing memory by ~4× for those entries.

During attention, cold entries are dequantised on-the-fly.

Reference: https://arxiv.org/abs/2502.10424
"""

import torch
import torch.nn as nn
import math


class QuantizedTensor:
    """Container for a symmetrically quantised INT8 tensor."""

    __slots__ = ("data_int8", "scale", "zero_point", "shape")

    def __init__(self, data_int8: torch.Tensor, scale: torch.Tensor,
                 zero_point: torch.Tensor, shape: tuple):
        self.data_int8 = data_int8
        self.scale = scale
        self.zero_point = zero_point
        self.shape = shape

    def dequantize(self) -> torch.Tensor:
        """Recover FP32/FP16 tensor."""
        return (self.data_int8.float() - self.zero_point.float()) * self.scale.float()

    @staticmethod
    def quantize(tensor: torch.Tensor) -> "QuantizedTensor":
        """Per-channel symmetric INT8 quantisation along last dim."""
        flat = tensor.reshape(-1, tensor.shape[-1])
        vmin = flat.min(dim=0).values
        vmax = flat.max(dim=0).values
        scale = (vmax - vmin) / 255.0
        scale = scale.clamp(min=1e-8)
        zero_point = (-vmin / scale).round().clamp(0, 255)

        quantized = ((flat / scale) + zero_point).round().clamp(0, 255).to(torch.uint8)
        return QuantizedTensor(
            data_int8=quantized.view(tensor.shape),
            scale=scale,
            zero_point=zero_point,
            shape=tensor.shape,
        )

    @property
    def device(self):
        return self.data_int8.device

    def to(self, device):
        return QuantizedTensor(
            self.data_int8.to(device), self.scale.to(device),
            self.zero_point.to(device), self.shape,
        )


class HierarchicalQuantizedCache(nn.Module):
    """
    Two-tier KV cache with automatic hot/cold partitioning.

    Tokens within ``hot_window`` of the current position stay in FP16/32.
    Older tokens are quantised to INT8 and dequantised on read.
    """

    def __init__(self, hot_window: int = 128):
        super().__init__()
        self.hot_window = hot_window
        self._hot: torch.Tensor | None = None
        self._cold: QuantizedTensor | None = None
        self._total_len: int = 0

    def reset(self):
        self._hot = None
        self._cold = None
        self._total_len = 0

    def append(self, new_entries: torch.Tensor):
        """
        Append new latent entries [B, S_new, D] to the cache.
        Automatically promotes overflow from hot → cold.
        """
        if self._hot is None:
            self._hot = new_entries
        else:
            self._hot = torch.cat([self._hot, new_entries], dim=1)

        self._total_len += new_entries.shape[1]

        # Move overflow to cold tier
        if self._hot.shape[1] > self.hot_window:
            n_to_cold = self._hot.shape[1] - self.hot_window
            to_cold = self._hot[:, :n_to_cold].detach()
            self._hot = self._hot[:, n_to_cold:]

            if self._cold is not None:
                existing = self._cold.dequantize().to(to_cold.device)
                combined = torch.cat([existing, to_cold], dim=1)
                self._cold = QuantizedTensor.quantize(combined)
            else:
                self._cold = QuantizedTensor.quantize(to_cold)

    def get_full_cache(self) -> torch.Tensor:
        """Return the full cache as a single FP tensor (cold is dequantised)."""
        parts = []
        if self._cold is not None:
            parts.append(self._cold.dequantize().to(self._hot.device))
        if self._hot is not None:
            parts.append(self._hot)

        if not parts:
            return None
        return torch.cat(parts, dim=1)

    @property
    def length(self) -> int:
        return self._total_len

    def memory_saved_ratio(self) -> float:
        """Estimate memory savings vs full FP16 storage."""
        if self._total_len == 0:
            return 1.0
        cold_len = self._cold.shape[0] if self._cold is not None else 0
        hot_len = self._hot.shape[1] if self._hot is not None else 0
        # INT8 = 1 byte, FP16 = 2 bytes → cold saves 50%
        total_fp16 = (cold_len + hot_len) * 2
        actual = cold_len * 1 + hot_len * 2
        return actual / max(total_fp16, 1)

