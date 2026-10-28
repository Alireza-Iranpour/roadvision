import torch

from roadvision.data.geometry import BEVGrid
from roadvision.models.backbone import BackboneConfig
from roadvision.models.bev_encoder import BEVEncoderConfig
from roadvision.models.depth import DepthHeadConfig
from roadvision.models.heads.detection_head import DetectionHeadConfig
from roadvision.models.heads.segmentation_head import SegmentationHeadConfig
from roadvision.models.roadvision_net import RoadVisionNet, RoadVisionNetConfig


def _tiny_config() -> RoadVisionNetConfig:
    return RoadVisionNetConfig(
        image_size=(64, 64),
        feature_size=(4, 4),
        dbound=(4.0, 12.0, 4.0),  # 2 bins
        bev_grid=BEVGrid(xbound=(-8.0, 8.0, 1.0), ybound=(-8.0, 8.0, 1.0)),
        backbone=BackboneConfig(name="resnet18", pretrained=False, out_channels=256),
        depth_head=DepthHeadConfig(num_depth_bins=2, feature_channels=16),
        bev_encoder=BEVEncoderConfig(name="resnet_bev", in_channels=16, out_channels=32, num_blocks=[1, 1]),
        segmentation_head=SegmentationHeadConfig(in_channels=32, num_classes=2, hidden_channels=8),
        detection_head=DetectionHeadConfig(in_channels=32, num_classes=10, hidden_channels=8),
    )


class TestRoadVisionNetForward:
    def test_output_shapes(self):
        config = _tiny_config()
        model = RoadVisionNet(config)
        model.eval()

        batch_size, num_cams = 2, 3
        batch = {
            "images": torch.randn(batch_size, num_cams, 3, 64, 64),
            "intrinsics": torch.tensor([[8.0, 0.0, 32.0], [0.0, 8.0, 32.0], [0.0, 0.0, 1.0]])
            .view(1, 1, 3, 3)
            .expand(batch_size, num_cams, 3, 3)
            .clone(),
            "cam_to_ego": torch.eye(4).view(1, 1, 4, 4).expand(batch_size, num_cams, 4, 4).clone(),
        }

        with torch.no_grad():
            out = model(batch)

        grid = config.bev_grid
        assert out["depth_dist"].shape == (batch_size, num_cams, 2, 4, 4)
        assert torch.allclose(out["depth_dist"].sum(dim=2), torch.ones(batch_size, num_cams, 4, 4), atol=1e-5)
        assert out["bev_features"].shape[0] == batch_size
        assert out["bev_features"].shape[1] == 32
        assert out["bev_features"].shape[2] <= grid.ny
        assert out["bev_features"].shape[3] <= grid.nx

        bev_h, bev_w = out["bev_features"].shape[2:]
        assert out["seg_logits"].shape == (batch_size, 2, bev_h, bev_w)
        assert out["det"]["heatmap"].shape == (batch_size, 10, bev_h, bev_w)
        assert out["det"]["offset"].shape == (batch_size, 2, bev_h, bev_w)
        assert out["det"]["dims"].shape == (batch_size, 2, bev_h, bev_w)
        assert out["det"]["yaw"].shape == (batch_size, 2, bev_h, bev_w)
        assert out["det"]["velocity"].shape == (batch_size, 2, bev_h, bev_w)
