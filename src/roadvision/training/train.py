"""Hydra entrypoint: `python -m roadvision.training.train` (single process)
or `torchrun --nproc_per_node=N -m roadvision.training.train training.distributed.enabled=true`.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler

from roadvision.data.geometry import BEVGrid
from roadvision.data.nuscenes_dataset import NuScenesBEVDataset
from roadvision.data.transforms import AugmentationConfig, CameraAugmentation, TransformedBEVDataset, collate_bev_batch
from roadvision.losses import DepthLoss, DetectionLoss, MultiTaskUncertaintyLoss, SegmentationLoss
from roadvision.models.roadvision_net import RoadVisionNet
from roadvision.training import engine
from roadvision.training.config import build_model_config, output_bev_grid
from roadvision.training.trackers import NullTracker, WandbTracker


def build_dataloaders(cfg: DictConfig, output_grid: BEVGrid, ctx: engine.DistributedContext) -> tuple[DataLoader, DataLoader]:
    data = cfg.data
    cameras = tuple(data.cameras)

    train_scenes = tuple(data.scene_split.train_scenes) if "scene_split" in data else None
    val_scenes = tuple(data.scene_split.val_scenes) if "scene_split" in data else None

    train_base = NuScenesBEVDataset(
        data.dataroot, version=data.version, cameras=cameras, output_grid=output_grid, scene_names=train_scenes
    )
    train_aug = CameraAugmentation(
        final_dim=tuple(data.image.final_dim), config=_augmentation_config(data), train=True
    )
    train_dataset = TransformedBEVDataset(train_base, train_aug, cameras)

    val_base = NuScenesBEVDataset(
        data.dataroot, version=data.version, cameras=cameras, output_grid=output_grid, scene_names=val_scenes
    )
    val_aug = CameraAugmentation(final_dim=tuple(data.image.final_dim), config=_augmentation_config(data), train=False)
    val_dataset = TransformedBEVDataset(val_base, val_aug, cameras)

    train_sampler = DistributedSampler(train_dataset) if ctx.enabled else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=data.dataloader.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=data.dataloader.num_workers,
        pin_memory=data.dataloader.pin_memory,
        collate_fn=collate_bev_batch,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data.dataloader.batch_size,
        shuffle=False,
        num_workers=data.dataloader.num_workers,
        pin_memory=data.dataloader.pin_memory,
        collate_fn=collate_bev_batch,
    )
    return train_loader, val_loader


def _augmentation_config(data: DictConfig) -> AugmentationConfig:
    aug = data.augmentation
    return AugmentationConfig(
        resize_lim=tuple(aug.resize_lim),
        crop_jitter_px=aug.crop_jitter_px,
        brightness=aug.photometric_jitter.brightness,
        contrast=aug.photometric_jitter.contrast,
        saturation=aug.photometric_jitter.saturation,
        hue=aug.photometric_jitter.hue,
        horizontal_flip_prob=aug.horizontal_flip_prob,
    )


def make_compute_losses(cfg: DictConfig) -> engine.ComputeLosses:
    depth_loss_fn = DepthLoss(tuple(cfg.data.depth.dbound), backbone_downsample=cfg.data.image.backbone_downsample)
    segmentation_loss_fn = SegmentationLoss()
    detection_loss_fn = DetectionLoss(per_class_normalize=cfg.model.detection_head.per_class_loss_normalize)

    def compute_losses(batch: dict[str, Any], model: nn.Module) -> dict[str, torch.Tensor]:
        out = model(batch)
        return {
            "depth": depth_loss_fn(out["depth_dist"], batch["depth_maps"]),
            "segmentation": segmentation_loss_fn(out["seg_logits"], batch["seg_target"]),
            "detection": detection_loss_fn(out["det"], batch["det_target"]),
        }

    return compute_losses


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)

    ctx = engine.setup_distributed(cfg.training.distributed.enabled, backend=cfg.training.distributed.backend)
    device = torch.device(f"cuda:{ctx.local_rank}" if torch.cuda.is_available() else "cpu")

    output_grid = output_bev_grid(cfg)
    train_loader, val_loader = build_dataloaders(cfg, output_grid, ctx)

    model = RoadVisionNet(build_model_config(cfg)).to(device)
    model = engine.wrap_model_for_distributed(model, ctx)
    multitask_loss = MultiTaskUncertaintyLoss(tuple(cfg.model.loss_weighting.tasks)).to(device)
    multitask_loss = engine.wrap_model_for_distributed(multitask_loss, ctx)

    optimizer = AdamW(
        itertools.chain(model.parameters(), multitask_loss.parameters()),
        lr=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.epochs * len(train_loader))
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.training.amp and device.type == "cuda")

    tracker = (
        WandbTracker(
            project=cfg.training.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.training.logging.mode,
            entity=cfg.training.logging.entity,
        )
        if cfg.training.logging.backend == "wandb" and ctx.is_main_process
        else NullTracker()
    )

    engine_config = engine.EngineConfig(
        epochs=cfg.training.epochs,
        grad_accumulation_steps=cfg.training.grad_accumulation_steps,
        amp=cfg.training.amp,
        checkpoint_dir=cfg.training.checkpoint.dir,
        save_every_n_epochs=cfg.training.checkpoint.save_every_n_epochs,
        keep_last_n=cfg.training.checkpoint.keep_last_n,
        log_every_n_steps=cfg.training.logging.log_every_n_steps,
    )
    compute_losses = make_compute_losses(cfg)
    state = engine.TrainState()
    checkpoint_dir = Path(engine_config.checkpoint_dir)

    for epoch in range(engine_config.epochs):
        state.epoch = epoch
        train_loss = engine.train_one_epoch(
            model, multitask_loss, train_loader, compute_losses, optimizer, scheduler, scaler, device,
            engine_config, state, tracker, ctx,
        )
        val_loss = engine.validate(model, multitask_loss, val_loader, compute_losses, device)

        if ctx.is_main_process:
            tracker.log_scalars({"epoch/train_loss": train_loss, "epoch/val_loss": val_loss}, step=state.global_step)
            if (epoch + 1) % engine_config.save_every_n_epochs == 0:
                engine.save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch:04d}.pt", model, multitask_loss, optimizer, scheduler, state
                )
                engine.prune_old_checkpoints(checkpoint_dir, engine_config.keep_last_n)

    tracker.finish()


if __name__ == "__main__":
    main()
