"""Small ResNet-style encoder refining the pooled BEV feature grid.

Reuses torchvision's `BasicBlock` rather than hand-rolling residual blocks.
Each of the `num_blocks` stages after the first halves spatial resolution
(stride 2), so a 200x200 input grid with `num_blocks=[2,2,2]` comes out at
50x50 -- downsampled but still dense enough for the task heads to place a
~0.5m-resolution grid cell's worth of detail per output cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models.resnet import BasicBlock


@dataclass(frozen=True)
class BEVEncoderConfig:
    name: str
    in_channels: int
    out_channels: int
    num_blocks: list[int]


def _make_stage(in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
    downsample = None
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    blocks = [BasicBlock(in_channels, out_channels, stride=stride, downsample=downsample)]
    blocks += [BasicBlock(out_channels, out_channels) for _ in range(num_blocks - 1)]
    return nn.Sequential(*blocks)


class BEVEncoder(nn.Module):
    def __init__(self, config: BEVEncoderConfig):
        super().__init__()
        widths = [config.in_channels * (2**i) for i in range(len(config.num_blocks))]
        widths[-1] = config.out_channels

        stages = []
        in_channels = config.in_channels
        for stage_idx, (width, depth) in enumerate(zip(widths, config.num_blocks)):
            stride = 1 if stage_idx == 0 else 2
            stages.append(_make_stage(in_channels, width, depth, stride))
            in_channels = width
        self.stages = nn.Sequential(*stages)
        self.out_channels = config.out_channels

    def forward(self, bev_features: torch.Tensor) -> torch.Tensor:
        return self.stages(bev_features)
