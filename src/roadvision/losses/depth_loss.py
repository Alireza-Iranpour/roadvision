"""Depth-distribution supervision from sparse LIDAR-projected depth.

`depth_dist` is predicted at backbone-feature resolution (stride 16) but
the LIDAR depth maps from `nuscenes_dataset.py` are full image resolution,
with 0.0 marking "no LIDAR return". Downsampling takes, per feature-stride
block, the *minimum* valid (non-zero) depth rather than an average: a block
straddling a foreground object's silhouette should take the nearer depth,
not a blend that belongs to neither surface. Blocks with no valid pixel are
masked out of the loss entirely, not pulled toward some default depth.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _downsample_sparse_depth(depth_maps: torch.Tensor, stride: int) -> tuple[torch.Tensor, torch.Tensor]:
    """depth_maps: (..., H, W) -> (nearest-valid depth, valid mask), both (..., H // stride, W // stride).

    Requires H, W divisible by `stride` (true for `image.final_dim` and
    `image.backbone_downsample` in the data configs).
    """
    *leading, height, width = depth_maps.shape
    if height % stride or width % stride:
        raise ValueError(f"depth map size ({height}, {width}) not divisible by stride {stride}")
    fh, fw = height // stride, width // stride

    blocks = depth_maps.reshape(*leading, fh, stride, fw, stride).permute(*range(len(leading)), -4, -2, -3, -1)
    blocks = blocks.reshape(*leading, fh, fw, stride * stride)  # (..., fh, fw, stride^2)

    valid = blocks > 0
    any_valid = valid.any(dim=-1)
    sentinel = torch.where(valid, blocks, blocks.new_full((), float("inf")))
    nearest = sentinel.min(dim=-1).values
    nearest = torch.where(any_valid, nearest, nearest.new_zeros(()))
    return nearest, any_valid


def depth_to_bin_index(depth: torch.Tensor, dbound: tuple[float, float, float], num_bins: int) -> torch.Tensor:
    """Nearest discretized depth-bin index for each depth value (clamped to
    the valid bin range -- callers must mask out invalid depths separately,
    this does not produce an "invalid" sentinel index)."""
    min_depth, _, step = dbound
    index = torch.round((depth - min_depth) / step).long()
    return index.clamp(0, num_bins - 1)


class DepthLoss(nn.Module):
    """Masked cross-entropy between predicted per-bin probabilities and the
    nearest discretized LIDAR depth bin, at valid (LIDAR-hit) pixels only."""

    def __init__(self, dbound: tuple[float, float, float], backbone_downsample: int, eps: float = 1e-6):
        super().__init__()
        self.dbound = dbound
        self.backbone_downsample = backbone_downsample
        self.eps = eps

    def forward(self, depth_dist: torch.Tensor, depth_maps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth_dist: (B, N, D, fH, fW) predicted per-bin probabilities (sums to 1 over D).
            depth_maps: (B, N, H, W) sparse LIDAR depth, 0.0 = no return, H = fH * stride.

        Returns: scalar loss, 0.0 (not NaN) if no pixel in the batch has a valid LIDAR depth.
        """
        num_bins = depth_dist.shape[2]
        nearest_depth, valid = _downsample_sparse_depth(depth_maps, self.backbone_downsample)
        if not valid.any():
            return depth_dist.sum() * 0.0

        target_bins = depth_to_bin_index(nearest_depth, self.dbound, num_bins)  # (B, N, fH, fW)
        log_probs = torch.log(depth_dist.clamp_min(self.eps))  # (B, N, D, fH, fW)

        per_pixel_nll = F.nll_loss(
            log_probs.permute(0, 1, 3, 4, 2).reshape(-1, num_bins),
            target_bins.reshape(-1),
            reduction="none",
        ).reshape(target_bins.shape)

        return (per_pixel_nll * valid).sum() / valid.sum().clamp_min(1)
