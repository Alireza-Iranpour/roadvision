"""Assembles backbone -> depth head -> BEV pool -> BEV encoder -> task
heads into the full multi-camera-images -> multi-task-predictions model.

The BEV encoder downsamples the pooled grid (e.g. 200x200 -> 50x50 with the
default `num_blocks=[2,2,2]`), so `seg_logits` and `det` predictions live on
a coarser grid than `configs/*/bev_grid`. Callers that rasterize ground
truth (training code, not this module) must use an "output grid" at that
same coarser resolution -- same xbound/ybound extent, resolution scaled by
the encoder's total stride -- when calling `bev_rasterize.*`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from roadvision.data.geometry import BEVGrid, make_depth_bins
from roadvision.models.backbone import BackboneConfig, ImageBackbone
from roadvision.models.bev_encoder import BEVEncoder, BEVEncoderConfig
from roadvision.models.bev_pool import BEVPool, FrustumConfig
from roadvision.models.depth import DepthHead, DepthHeadConfig
from roadvision.models.heads.detection_head import DetectionHead, DetectionHeadConfig
from roadvision.models.heads.segmentation_head import SegmentationHead, SegmentationHeadConfig


@dataclass(frozen=True)
class RoadVisionNetConfig:
    image_size: tuple[int, int]  # final_dim (H, W)
    feature_size: tuple[int, int]  # backbone output (fH, fW)
    dbound: tuple[float, float, float]
    bev_grid: BEVGrid
    backbone: BackboneConfig
    depth_head: DepthHeadConfig
    bev_encoder: BEVEncoderConfig
    segmentation_head: SegmentationHeadConfig
    detection_head: DetectionHeadConfig


class RoadVisionNet(nn.Module):
    def __init__(self, config: RoadVisionNetConfig):
        super().__init__()
        self.backbone = ImageBackbone(config.backbone)
        self.depth_head = DepthHead(config.backbone.out_channels, config.depth_head)

        depth_bins = torch.from_numpy(make_depth_bins(config.dbound)).float()
        if depth_bins.shape[0] != config.depth_head.num_depth_bins:
            raise ValueError(
                f"dbound {config.dbound} produces {depth_bins.shape[0]} bins, "
                f"but depth_head.num_depth_bins={config.depth_head.num_depth_bins}"
            )
        frustum_config = FrustumConfig(image_size=config.image_size, feature_size=config.feature_size)
        self.bev_pool = BEVPool(frustum_config, depth_bins, config.bev_grid)

        self.bev_encoder = BEVEncoder(config.bev_encoder)
        self.segmentation_head = SegmentationHead(config.segmentation_head)
        self.detection_head = DetectionHead(config.detection_head)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        images = batch["images"]  # (B, N, 3, H, W)
        intrinsics = batch["intrinsics"]  # (B, N, 3, 3)
        cam_to_ego = batch["cam_to_ego"]  # (B, N, 4, 4)
        batch_size, num_cams = images.shape[:2]

        flat_images = images.reshape(batch_size * num_cams, *images.shape[2:])
        features = self.backbone(flat_images)
        depth_dist, context = self.depth_head(features)

        feat_h, feat_w = depth_dist.shape[-2:]
        num_bins = depth_dist.shape[1]
        num_context = context.shape[1]
        depth_dist = depth_dist.view(batch_size, num_cams, num_bins, feat_h, feat_w)
        context = context.view(batch_size, num_cams, num_context, feat_h, feat_w)

        bev_features = self.bev_pool(depth_dist, context, intrinsics, cam_to_ego)
        bev_features = self.bev_encoder(bev_features)

        seg_logits = self.segmentation_head(bev_features)
        det = self.detection_head(bev_features)

        return {
            "bev_features": bev_features,
            "depth_dist": depth_dist,
            "seg_logits": seg_logits,
            "det": det,
        }
