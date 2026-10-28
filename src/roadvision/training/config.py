"""Converts the Hydra/OmegaConf `cfg` tree into the plain dataclasses the
model/data modules take -- keeps every other module ignorant of Hydra and
testable with hand-built dataclasses (see `tests/test_roadvision_net.py`).
"""

from __future__ import annotations

from omegaconf import DictConfig

from roadvision.data.geometry import BEVGrid
from roadvision.models.backbone import BackboneConfig
from roadvision.models.bev_encoder import BEVEncoderConfig
from roadvision.models.depth import DepthHeadConfig
from roadvision.models.heads.detection_head import DetectionHeadConfig
from roadvision.models.heads.segmentation_head import SegmentationHeadConfig
from roadvision.models.roadvision_net import RoadVisionNetConfig


def build_model_config(cfg: DictConfig) -> RoadVisionNetConfig:
    data, model = cfg.data, cfg.model
    return RoadVisionNetConfig(
        image_size=tuple(data.image.final_dim),
        feature_size=output_feature_size(data),
        dbound=tuple(data.depth.dbound),
        bev_grid=BEVGrid(xbound=tuple(data.bev_grid.xbound), ybound=tuple(data.bev_grid.ybound)),
        backbone=BackboneConfig(
            name=model.backbone.name, pretrained=model.backbone.pretrained, out_channels=model.backbone.out_channels
        ),
        depth_head=DepthHeadConfig(
            num_depth_bins=model.depth_head.num_depth_bins, feature_channels=model.depth_head.feature_channels
        ),
        bev_encoder=BEVEncoderConfig(
            name=model.bev_encoder.name,
            in_channels=model.bev_encoder.in_channels,
            out_channels=model.bev_encoder.out_channels,
            num_blocks=list(model.bev_encoder.num_blocks),
        ),
        segmentation_head=SegmentationHeadConfig(
            in_channels=model.bev_encoder.out_channels,
            num_classes=model.segmentation_head.num_classes,
            hidden_channels=model.segmentation_head.hidden_channels,
        ),
        detection_head=DetectionHeadConfig(
            in_channels=model.bev_encoder.out_channels,
            num_classes=model.detection_head.num_classes,
            hidden_channels=model.detection_head.hidden_channels,
        ),
    )


def output_feature_size(data: DictConfig) -> tuple[int, int]:
    final_h, final_w = data.image.final_dim
    stride = data.image.backbone_downsample
    return final_h // stride, final_w // stride


def output_bev_grid(cfg: DictConfig) -> BEVGrid:
    """The BEV grid at the *model output* resolution (after the BEV
    encoder's downsampling) -- used to rasterize detection/segmentation
    ground truth so targets line up with `seg_logits`/`det` predictions.
    See `roadvision_net.py`'s module docstring for why these differ from
    `data.bev_grid`.
    """
    data, model = cfg.data, cfg.model
    downsample = _encoder_downsample(model.bev_encoder.num_blocks)
    xmin, xmax, xres = data.bev_grid.xbound
    ymin, ymax, yres = data.bev_grid.ybound
    return BEVGrid(xbound=(xmin, xmax, xres * downsample), ybound=(ymin, ymax, yres * downsample))


def _encoder_downsample(num_blocks: list[int]) -> int:
    """`BEVEncoder` applies stride 1 to the first stage and stride 2 to
    every stage after it (see `bev_encoder.py`)."""
    return 2 ** max(len(num_blocks) - 1, 0)
