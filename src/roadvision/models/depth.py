"""Per-camera depth-distribution head.

Standard LSS design: a single conv produces `num_depth_bins + context_channels`
channels per pixel. The first `num_depth_bins` channels are softmax-normalized
over the depth axis (a categorical "how likely is this pixel at each depth
bin" distribution); the remaining `context_channels` are an unnormalized
per-pixel feature vector. `bev_pool` combines them by outer product so every
frustum point gets `context * P(depth bin)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DepthHeadConfig:
    num_depth_bins: int
    feature_channels: int


class DepthHead(nn.Module):
    def __init__(self, in_channels: int, config: DepthHeadConfig):
        super().__init__()
        self.num_depth_bins = config.num_depth_bins
        self.feature_channels = config.feature_channels
        out_channels = config.num_depth_bins + config.feature_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """features: (B, in_channels, H, W).

        Returns:
            depth_dist: (B, num_depth_bins, H, W), softmax over dim=1.
            context: (B, feature_channels, H, W).
        """
        out = self.conv(features)
        depth_logits, context = out.split([self.num_depth_bins, self.feature_channels], dim=1)
        depth_dist = depth_logits.softmax(dim=1)
        return depth_dist, context
