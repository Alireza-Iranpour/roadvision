"""End-to-end smoke test standing in for a real training run: a tiny
synthetic in-memory dataset (random tensors matching the real batch schema)
run through the actual training engine, on CPU, with DDP and W&B both
disabled. This is the only check in this environment that the
forward -> multitask-loss -> backward -> optimizer-step -> checkpoint
round trip actually works together, since real nuScenes-mini data isn't
downloaded here.
"""

from __future__ import annotations

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from roadvision.data.geometry import BEVGrid
from roadvision.losses import DepthLoss, DetectionLoss, MultiTaskUncertaintyLoss, SegmentationLoss
from roadvision.models.backbone import BackboneConfig
from roadvision.models.bev_encoder import BEVEncoderConfig
from roadvision.models.depth import DepthHeadConfig
from roadvision.models.heads.detection_head import DetectionHeadConfig
from roadvision.models.heads.segmentation_head import SegmentationHeadConfig
from roadvision.models.roadvision_net import RoadVisionNet, RoadVisionNetConfig
from roadvision.training import engine

_BATCH_SIZE, _NUM_CAMS, _IMG_H, _IMG_W = 2, 2, 64, 64
_DBOUND = (4.0, 12.0, 4.0)  # 2 bins
_BEV_GRID = BEVGrid(xbound=(-8.0, 8.0, 1.0), ybound=(-8.0, 8.0, 1.0))  # 16x16
_OUT_H, _OUT_W = 8, 8  # BEVEncoder with num_blocks=[1, 1] halves once: 16 -> 8
_NUM_DET_CLASSES = 3


class _SyntheticBEVDataset(Dataset):
    """Random tensors matching exactly the keys/shapes `collate_bev_batch`
    would produce for a real nuScenes sample at this test's tiny config."""

    def __init__(self, length: int):
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        torch.manual_seed(index)
        intrinsic = torch.tensor([[32.0, 0.0, 32.0], [0.0, 32.0, 32.0], [0.0, 0.0, 1.0]])
        heatmap = torch.zeros(_NUM_DET_CLASSES, _OUT_H, _OUT_W)
        heatmap[0, 4, 4] = 1.0
        mask = torch.zeros(_OUT_H, _OUT_W)
        mask[4, 4] = 1.0
        return {
            "images": torch.randn(_NUM_CAMS, 3, _IMG_H, _IMG_W),
            "intrinsics": intrinsic.unsqueeze(0).expand(_NUM_CAMS, 3, 3).clone(),
            "cam_to_ego": torch.eye(4).unsqueeze(0).expand(_NUM_CAMS, 4, 4).clone(),
            "depth_maps": torch.randint(0, 2, (_NUM_CAMS, _IMG_H, _IMG_W)).float() * 8.0,
            "seg_target": (torch.rand(2, _OUT_H, _OUT_W) > 0.5).float(),
            "det_target": {
                "heatmap": heatmap,
                "offset": torch.zeros(2, _OUT_H, _OUT_W),
                "dims": torch.zeros(2, _OUT_H, _OUT_W),
                "yaw": torch.zeros(2, _OUT_H, _OUT_W),
                "velocity": torch.zeros(2, _OUT_H, _OUT_W),
                "mask": mask,
            },
        }


def _collate(samples: list[dict]) -> dict:
    batch = {
        key: torch.stack([s[key] for s in samples], dim=0) for key in ("images", "intrinsics", "cam_to_ego", "depth_maps", "seg_target")
    }
    batch["det_target"] = {
        key: torch.stack([s["det_target"][key] for s in samples], dim=0) for key in samples[0]["det_target"]
    }
    return batch


def _tiny_model() -> RoadVisionNet:
    config = RoadVisionNetConfig(
        image_size=(_IMG_H, _IMG_W),
        feature_size=(4, 4),
        dbound=_DBOUND,
        bev_grid=_BEV_GRID,
        backbone=BackboneConfig(name="resnet18", pretrained=False, out_channels=256),
        depth_head=DepthHeadConfig(num_depth_bins=2, feature_channels=16),
        bev_encoder=BEVEncoderConfig(name="resnet_bev", in_channels=16, out_channels=32, num_blocks=[1, 1]),
        segmentation_head=SegmentationHeadConfig(in_channels=32, num_classes=2, hidden_channels=8),
        detection_head=DetectionHeadConfig(in_channels=32, num_classes=_NUM_DET_CLASSES, hidden_channels=8),
    )
    return RoadVisionNet(config)


def _compute_losses(depth_loss_fn, seg_loss_fn, det_loss_fn):
    def compute(batch: dict, model) -> dict[str, torch.Tensor]:
        out = model(batch)
        return {
            "depth": depth_loss_fn(out["depth_dist"], batch["depth_maps"]),
            "segmentation": seg_loss_fn(out["seg_logits"], batch["seg_target"]),
            "detection": det_loss_fn(out["det"], batch["det_target"]),
        }

    return compute


class TestTrainingSmoke:
    def test_train_one_epoch_runs_and_checkpoint_round_trips(self, tmp_path):
        torch.manual_seed(0)
        model = _tiny_model()
        multitask_loss = MultiTaskUncertaintyLoss()
        device = torch.device("cpu")

        dataset = _SyntheticBEVDataset(length=4)
        dataloader = DataLoader(dataset, batch_size=_BATCH_SIZE, collate_fn=_collate)

        depth_loss_fn = DepthLoss(_DBOUND, backbone_downsample=16)
        compute_losses = _compute_losses(depth_loss_fn, SegmentationLoss(), DetectionLoss())

        optimizer = AdamW(list(model.parameters()) + list(multitask_loss.parameters()), lr=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        config = engine.EngineConfig(
            epochs=1,
            grad_accumulation_steps=1,
            amp=False,
            checkpoint_dir=str(tmp_path),
            save_every_n_epochs=1,
            keep_last_n=1,
            log_every_n_steps=1,
        )
        state = engine.TrainState()
        ctx = engine.DistributedContext(enabled=False)
        tracker = _RecordingTracker()

        first_loss = engine.train_one_epoch(
            model, multitask_loss, dataloader, compute_losses, optimizer, scheduler, scaler, device, config, state,
            tracker, ctx,
        )
        assert torch.isfinite(torch.tensor(first_loss))
        assert state.global_step == len(dataloader)
        assert len(tracker.logged) > 0

        val_loss = engine.validate(model, multitask_loss, dataloader, compute_losses, device)
        assert torch.isfinite(torch.tensor(val_loss))

        checkpoint_path = tmp_path / "epoch_0000.pt"
        engine.save_checkpoint(checkpoint_path, model, multitask_loss, optimizer, scheduler, state)
        assert checkpoint_path.exists()

        restored_model = _tiny_model()
        restored_loss = MultiTaskUncertaintyLoss()
        restored_optimizer = AdamW(list(restored_model.parameters()) + list(restored_loss.parameters()), lr=1e-3)
        restored_state = engine.load_checkpoint(checkpoint_path, restored_model, restored_loss, restored_optimizer)
        assert restored_state.epoch == state.epoch
        assert restored_state.global_step == state.global_step
        for p1, p2 in zip(model.parameters(), restored_model.parameters()):
            assert torch.allclose(p1, p2)


class _RecordingTracker:
    def __init__(self):
        self.logged: list[dict] = []

    def log_scalars(self, scalars: dict, step: int) -> None:
        self.logged.append(scalars)

    def log_images(self, images: dict, step: int) -> None:
        pass

    def finish(self) -> None:
        pass
