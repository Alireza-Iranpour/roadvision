"""BEV detection loss: CenterNet-style penalty-reduced focal loss on the
heatmap, plus masked L1 regression on box parameters.

`bev_rasterize.rasterize_boxes_to_bev` already produces soft Gaussian
heatmap targets in [0, 1] (not hard 0/1 keypoints), so the "negative"
branch below uses the standard CornerNet/CenterNet `(1 - y)^beta` penalty
reduction near object centers -- a cell one pixel off-center is a much
softer negative than one far from any object, which keeps the loss from
punishing localization noise as hard as a true false positive.

Regression (`offset`, `dims`, `yaw`, `velocity`) is only meaningful at the
object-center cells `bev_rasterize` marks in `mask`; off-center cells have
no regression target at all (zeros is not "the answer", it's "unset"), so
those predictions must not contribute to the regression loss regardless of
their value.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def gaussian_focal_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
    eps: float = 1e-6,
    per_class_normalize: bool = True,
) -> torch.Tensor:
    """
    Args:
        pred_logits: (B, num_classes, H, W) raw heatmap logits.
        target: (B, num_classes, H, W) Gaussian-splatted targets in [0, 1], 1.0 at exact object centers.
        per_class_normalize: if True (default), each class's positive+negative
            loss is normalized by *that class's own* center count before
            averaging across classes, rather than one global count pooled
            over all classes. nuScenes' real class frequencies are wildly
            skewed (~7600 `car` annotations vs ~60 `trailer` in v1.0-mini) --
            with a single global count, a rare class's handful of centers
            contribute a vanishingly small fraction of the gradient relative
            to its own difficulty, and the loss is effectively dominated by
            whichever class happens to be most common in each batch. Per-
            class normalization gives every class a comparable per-center
            gradient magnitude regardless of how often it appears.

    Returns: scalar loss.
    """
    pred = torch.sigmoid(pred_logits).clamp(eps, 1 - eps)
    is_center = target.eq(1.0).float()

    positive_loss = -((1 - pred) ** alpha) * torch.log(pred) * is_center
    negative_loss = -((1 - target) ** beta) * (pred**alpha) * torch.log(1 - pred) * (1 - is_center)
    per_pixel_loss = positive_loss + negative_loss

    if not per_class_normalize:
        num_centers = is_center.sum().clamp_min(1)
        return per_pixel_loss.sum() / num_centers

    num_classes = target.shape[1]
    class_dims = tuple(d for d in range(target.dim()) if d != 1)
    centers_per_class = is_center.sum(dim=class_dims).clamp_min(1)  # (num_classes,)
    loss_per_class = per_pixel_loss.sum(dim=class_dims)  # (num_classes,)
    return (loss_per_class / centers_per_class).sum() / num_classes


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Args:
        pred, target: (B, C, H, W).
        mask: (B, H, W) or (B, 1, H, W), 1 at positions with a valid regression target.
    """
    if mask.dim() == pred.dim() - 1:
        mask = mask.unsqueeze(1)
    diff = (pred - target).abs() * mask
    num_valid = (mask.sum() * pred.shape[1]).clamp_min(1)
    return diff.sum() / num_valid


@dataclass(frozen=True)
class DetectionLossWeights:
    heatmap: float = 1.0
    offset: float = 1.0
    dims: float = 1.0
    yaw: float = 1.0
    velocity: float = 1.0


class DetectionLoss(nn.Module):
    def __init__(self, weights: DetectionLossWeights = DetectionLossWeights(), per_class_normalize: bool = True):
        super().__init__()
        self.weights = weights
        self.per_class_normalize = per_class_normalize

    def forward(self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            pred: dict with keys heatmap/offset/dims/yaw/velocity (model output, raw logits/regression).
            target: dict with the same keys plus `mask`, matching
                `roadvision.data.bev_rasterize.DetectionTargets` fields.
        """
        mask = target["mask"]
        w = self.weights
        return (
            w.heatmap * gaussian_focal_loss(pred["heatmap"], target["heatmap"], per_class_normalize=self.per_class_normalize)
            + w.offset * masked_l1_loss(pred["offset"], target["offset"], mask)
            + w.dims * masked_l1_loss(pred["dims"], target["dims"], mask)
            + w.yaw * masked_l1_loss(pred["yaw"], target["yaw"], mask)
            + w.velocity * masked_l1_loss(pred["velocity"], target["velocity"], mask)
        )
