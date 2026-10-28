"""Learned homoscedastic uncertainty weighting across the depth/segmentation
/detection losses (Kendall, Gal & Cipolla, 2018 -- "Multi-Task Learning
Using Uncertainty to Weigh Losses for Scene Geometry and Semantics").

Each task owns a learned log-variance `s_i = log(sigma_i^2)`, combined as
`total = sum_i exp(-s_i) * L_i + s_i`. A task whose loss is noisy/hard to
drive down pushes `s_i` up, which *shrinks* its own gradient contribution
(`exp(-s_i)` falls) while the `+ s_i` term stops it from inflating `s_i`
to infinity and checking out of training entirely. `s_i` (not sigma_i
directly) is the learned parameter so it can go negative without
constraint, and `exp(-s_i)` is always positive by construction -- no need
for e.g. a softplus to keep weights non-negative.
"""

from __future__ import annotations

import torch
from torch import nn


class MultiTaskUncertaintyLoss(nn.Module):
    def __init__(self, task_names: tuple[str, ...] = ("depth", "segmentation", "detection")):
        super().__init__()
        self.task_names = task_names
        self.log_vars = nn.Parameter(torch.zeros(len(task_names)))

    def forward(self, task_losses: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Args:
            task_losses: dict mapping each name in `task_names` to a scalar loss tensor.

        Returns: dict with `total` (the weighted sum to call `.backward()` on),
            `weighted/<name>` (that task's contribution to `total`), and
            `uncertainty/<name>` (the current learned sigma_i, for logging).
        """
        missing = set(self.task_names) - set(task_losses)
        if missing:
            raise ValueError(f"missing task losses: {sorted(missing)}")

        out: dict[str, torch.Tensor] = {}
        total = task_losses[self.task_names[0]].new_zeros(())
        for i, name in enumerate(self.task_names):
            log_var = self.log_vars[i]
            weighted = torch.exp(-log_var) * task_losses[name] + log_var
            out[f"weighted/{name}"] = weighted.detach()
            out[f"uncertainty/{name}"] = torch.exp(0.5 * log_var).detach()
            total = total + weighted

        out["total"] = total
        return out
