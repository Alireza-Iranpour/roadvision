"""Hydra entrypoint: `python -m roadvision.eval.evaluate training.checkpoint.resume_path=<path/to/epoch_XXXX.pt>`.

Mirrors `training/train.py`'s config wiring (same `build_model_config`,
`output_bev_grid`, dataset/dataloader construction) so a checkpoint trained
under one config evaluates under the matching one. This module has not been
run end-to-end in this environment -- there is no downloaded nuScenes-mini
data or trained checkpoint to run it against here. It is verified by
import-time correctness and by `eval/metrics.py` and `eval/failure_analysis.py`
each being independently unit-tested against synthetic data with hand-
computed expected values; the data-loading/inference loop below is
otherwise unverified until run against real data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from roadvision.data.geometry import BEVGrid
from roadvision.data.nuscenes_dataset import NuScenesBEVDataset
from roadvision.data.transforms import AugmentationConfig, CameraAugmentation, TransformedBEVDataset, collate_bev_batch
from roadvision.eval.failure_analysis import worst_k_samples
from roadvision.eval.metrics import (
    compute_depth_metrics,
    compute_segmentation_iou,
    expected_depth_from_distribution,
    match_detections_by_center_distance,
)
from roadvision.losses import MultiTaskUncertaintyLoss
from roadvision.losses.depth_loss import _downsample_sparse_depth
from roadvision.models.roadvision_net import RoadVisionNet
from roadvision.training import engine
from roadvision.training.config import build_model_config, output_bev_grid


def build_val_dataloader(cfg: DictConfig, output_grid: BEVGrid) -> DataLoader:
    data = cfg.data
    cameras = tuple(data.cameras)
    val_scenes = tuple(data.scene_split.val_scenes) if "scene_split" in data else None
    dataset = NuScenesBEVDataset(
        data.dataroot, version=data.version, cameras=cameras, output_grid=output_grid, scene_names=val_scenes
    )
    augmentation = CameraAugmentation(
        final_dim=tuple(data.image.final_dim),
        config=AugmentationConfig(
            resize_lim=tuple(data.augmentation.resize_lim),
            crop_jitter_px=data.augmentation.crop_jitter_px,
            brightness=data.augmentation.photometric_jitter.brightness,
            contrast=data.augmentation.photometric_jitter.contrast,
            saturation=data.augmentation.photometric_jitter.saturation,
            hue=data.augmentation.photometric_jitter.hue,
            horizontal_flip_prob=data.augmentation.horizontal_flip_prob,
        ),
        train=False,
    )
    transformed = TransformedBEVDataset(dataset, augmentation, cameras)
    return DataLoader(
        transformed,
        batch_size=data.dataloader.batch_size,
        shuffle=False,
        num_workers=data.dataloader.num_workers,
        collate_fn=collate_bev_batch,
    )


def gt_centers_for_class(det_target: dict[str, torch.Tensor], batch_idx: int, class_idx: int) -> torch.Tensor:
    """Ground-truth object centers (cell-unit xy) for one class, in one
    sample of a batch -- from that class's own heatmap channel (exact
    centers at value == 1.0), NOT `det_target["mask"]`, which marks every
    class's object centers in one shared channel and would silently credit
    a class's AP with other classes' objects if used here (a real bug this
    was extracted from, see Phase 5 notes in docs/RESULTS.md)."""
    gt_mask = det_target["heatmap"][batch_idx, class_idx].eq(1.0)
    ys, xs = torch.nonzero(gt_mask, as_tuple=True)
    offset = det_target["offset"][batch_idx]
    return torch.stack([xs.double() + offset[0, ys, xs], ys.double() + offset[1, ys, xs]], dim=-1)


@torch.no_grad()
def evaluate_dataset(
    model: RoadVisionNet, dataloader: DataLoader, dbound: tuple[float, float, float], backbone_downsample: int
) -> dict[str, Any]:
    """Runs inference over `dataloader`, accumulating per-sample metrics.
    Returns a results dict with batch-averaged metrics plus the worst-K
    sample tokens by depth AbsRel, ready for `failure_analysis.slice_metrics_by_attribute`
    once real scenario attributes are available for those tokens.
    """
    model.eval()
    depth_results, seg_results, det_results = [], [], []
    tokens: list[str] = []
    per_sample_abs_rel: list[float] = []

    for batch in dataloader:
        out = model(batch)

        pred_depth = expected_depth_from_distribution(out["depth_dist"], dbound)
        gt_depth, valid_mask = _downsample_sparse_depth(batch["depth_maps"], backbone_downsample)
        depth_metrics = compute_depth_metrics(pred_depth, gt_depth, valid_mask)
        depth_results.append(depth_metrics)

        seg_results.append(compute_segmentation_iou(out["seg_logits"], batch["seg_target"]))

        batch_size, num_classes = out["det"]["heatmap"].shape[:2]
        for b in range(batch_size):
            per_class_ap = []
            for c in range(num_classes):
                gt_centers = gt_centers_for_class(batch["det_target"], b, c)
                result = match_detections_by_center_distance(
                    torch.sigmoid(out["det"]["heatmap"][b, c]), out["det"]["offset"][b], gt_centers, distance_threshold=2.0
                )
                per_class_ap.append(result.ap)
            det_results.append(per_class_ap)

        tokens.extend(batch["tokens"])
        sample_abs_rel = (pred_depth - gt_depth).abs() / gt_depth.clamp_min(1e-6)
        sample_abs_rel = torch.where(valid_mask, sample_abs_rel, sample_abs_rel.new_zeros(()))
        valid_count = valid_mask.flatten(1).sum(dim=1).clamp_min(1)
        per_sample_abs_rel.extend((sample_abs_rel.flatten(1).sum(dim=1) / valid_count).tolist())

    worst_depth_samples = worst_k_samples(tokens, per_sample_abs_rel, k=min(20, len(tokens)))

    return {
        "num_samples": len(tokens),
        "depth": _average_dataclass_fields(depth_results),
        "segmentation": _average_dict_fields(seg_results),
        "detection_mean_ap": sum(ap for sample in det_results for ap in sample) / max(sum(len(s) for s in det_results), 1),
        "worst_depth_samples": worst_depth_samples,
        "tokens": tokens,
        "per_sample_depth_abs_rel": per_sample_abs_rel,
    }


def _average_dataclass_fields(items: list[Any]) -> dict[str, float]:
    if not items:
        return {}
    fields = items[0].__dataclass_fields__
    return {name: sum(getattr(item, name) for item in items) / len(items) for name in fields}


def _average_dict_fields(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = items[0].keys()
    return {key: sum(item[key] for item in items) / len(items) for key in keys}


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_grid = output_bev_grid(cfg)
    val_loader = build_val_dataloader(cfg, output_grid)

    model = RoadVisionNet(build_model_config(cfg)).to(device)
    multitask_loss = MultiTaskUncertaintyLoss(tuple(cfg.model.loss_weighting.tasks)).to(device)
    engine.load_checkpoint(Path(cfg.training.checkpoint.resume_path), model, multitask_loss, map_location=device)

    results = evaluate_dataset(model, val_loader, tuple(cfg.data.depth.dbound), cfg.data.image.backbone_downsample)

    output_path = Path(cfg.training.checkpoint.dir) / "eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
