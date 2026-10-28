"""Shared per-camera image backbone.

Truncated at the stride-16 stage (`layer3`) rather than the full stride-32
network: lift-splat needs a feature map dense enough to resolve individual
frustum columns, and stride-16 is the standard choice in LSS-family models.
ResNet-34's `layer3` outputs 256 channels; a 1x1 conv remaps that to the
configured `out_channels` so the rest of the model's channel width is a
free config choice rather than being pinned to the backbone's native width.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import resnet34, resnet18, ResNet34_Weights, ResNet18_Weights

_RESNET_LAYER3_CHANNELS = {"resnet18": 256, "resnet34": 256}  # both use BasicBlock's 64/128/256/512 stage widths
_RESNET_BUILDERS = {
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
    "resnet34": (resnet34, ResNet34_Weights.DEFAULT),
}


@dataclass(frozen=True)
class BackboneConfig:
    name: str
    pretrained: bool
    out_channels: int


class ImageBackbone(nn.Module):
    """Per-camera feature extractor, shared across all cameras.

    Input:  (B, 3, H, W)
    Output: (B, out_channels, H // 16, W // 16)
    """

    def __init__(self, config: BackboneConfig):
        super().__init__()
        if config.name not in _RESNET_BUILDERS:
            raise ValueError(f"unsupported backbone '{config.name}', expected one of {sorted(_RESNET_BUILDERS)}")
        builder, weights = _RESNET_BUILDERS[config.name]
        net = builder(weights=weights if config.pretrained else None)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3

        native_channels = _RESNET_LAYER3_CHANNELS[config.name]
        self.project = (
            nn.Identity()
            if config.out_channels == native_channels
            else nn.Conv2d(native_channels, config.out_channels, kernel_size=1)
        )
        self.out_channels = config.out_channels

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.project(x)
