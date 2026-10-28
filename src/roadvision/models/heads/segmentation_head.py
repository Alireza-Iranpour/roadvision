"""BEV semantic segmentation head.

Classes (`drivable_area`, `lane`, see `configs/model/roadvision_net.yaml`)
are not mutually exclusive -- a lane marking sits on top of drivable area
-- so this is multi-label (independent per-class sigmoid), not a softmax
over classes. `SegmentationLoss` (in `roadvision.losses`) expects raw
logits, one channel per class, matching `bev_rasterize.SEGMENTATION_LAYERS`
order.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SegmentationHeadConfig:
    in_channels: int
    num_classes: int
    hidden_channels: int = 128


class SegmentationHead(nn.Module):
    def __init__(self, config: SegmentationHeadConfig):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(config.in_channels, config.hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.hidden_channels, config.hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.hidden_channels, config.num_classes, kernel_size=1),
        )

    def forward(self, bev_features: torch.Tensor) -> torch.Tensor:
        """bev_features: (B, in_channels, H, W) -> logits (B, num_classes, H, W)."""
        return self.conv(bev_features)
