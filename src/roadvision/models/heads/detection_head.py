"""CenterPoint-style BEV detection head.

Output keys mirror `roadvision.data.bev_rasterize.DetectionTargets` field
names 1:1 (`heatmap`, `offset`, `dims`, `yaw`, `velocity`) so
`DetectionLoss` can zip predictions against targets without a translation
layer. Only what `bev_rasterize` actually supplies as ground truth is
regressed -- no height/z-center or full 3D extent, since nuScenes boxes are
rasterized to a 2D BEV footprint (center_x, center_y, length, width, yaw)
plus planar velocity.

Yaw is regressed as (sin, cos) rather than a raw angle to avoid the
angle-wraparound discontinuity at +-pi; `dims` are regressed in log-space
since box length/width are strictly positive and roughly log-normally
distributed across object classes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DetectionHeadConfig:
    in_channels: int
    num_classes: int
    hidden_channels: int = 128


def _conv_head(in_channels: int, hidden_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(hidden_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
    )


class DetectionHead(nn.Module):
    def __init__(self, config: DetectionHeadConfig):
        super().__init__()
        self.heatmap = _conv_head(config.in_channels, config.hidden_channels, config.num_classes)
        self.offset = _conv_head(config.in_channels, config.hidden_channels, 2)
        self.dims = _conv_head(config.in_channels, config.hidden_channels, 2)
        self.yaw = _conv_head(config.in_channels, config.hidden_channels, 2)
        self.velocity = _conv_head(config.in_channels, config.hidden_channels, 2)

        # CenterNet convention: initialize the heatmap branch's final-layer
        # bias so sigmoid(logit) starts near the true positive rate (most
        # cells are background), rather than 0.5 everywhere -- stabilizes
        # early focal-loss training.
        final_conv = self.heatmap[-1]
        nn.init.constant_(final_conv.bias, -2.19)  # sigmoid(-2.19) ~= 0.1

    def forward(self, bev_features: torch.Tensor) -> dict[str, torch.Tensor]:
        """bev_features: (B, in_channels, H, W).

        Returns raw logits for `heatmap` (sigmoid applied by the loss/by the
        caller at inference time, not here) and raw regression outputs for
        `offset`, `dims`, `yaw`, `velocity` -- all (B, ., H, W).
        """
        return {
            "heatmap": self.heatmap(bev_features),
            "offset": self.offset(bev_features),
            "dims": self.dims(bev_features),
            "yaw": self.yaw(bev_features),
            "velocity": self.velocity(bev_features),
        }
