"""BEV segmentation loss: per-class sigmoid focal loss (multi-label).

`drivable_area` covers most of the grid while `lane` is a thin minority
class, so a plain BCE would let the model coast by predicting
"mostly drivable, never lane". Per-class `pos_weight` corrects the
positive/negative *count* imbalance (computed from the batch itself, not
hardcoded, since the ratio depends on the BEV grid/crop and would silently
go stale if hand-tuned); focal loss's `gamma` additionally down-weights
already-easy (confidently correct) cells so training spends its gradient
budget on the hard boundary pixels, independent of the class imbalance
correction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SegmentationLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, eps: float = 1e-6):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, num_classes, H, W) raw logits.
            targets: (B, num_classes, H, W) binary (0/1) ground truth, per class.

        Returns: scalar loss, averaged over all elements.
        """
        num_positive = targets.sum(dim=(0, 2, 3))
        num_negative = targets.numel() / targets.shape[1] - num_positive
        pos_weight = (num_negative / num_positive.clamp_min(1)).clamp(max=100.0)
        pos_weight = pos_weight.view(1, -1, 1, 1)

        probs = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probs, 1 - probs).clamp_min(self.eps)
        focal_weight = (1 - pt) ** self.gamma

        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction="none"
        )
        return (focal_weight * bce).mean()
