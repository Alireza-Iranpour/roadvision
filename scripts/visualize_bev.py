"""Overlay BEV segmentation + detection predictions on the BEV grid and save
a figure, for visual inspection of model outputs (including on synthetic
data, so this is runnable without a downloaded dataset or trained model --
see `tests/test_visualize_bev.py`).

Usage: `python scripts/visualize_bev.py --checkpoint path/to/epoch_XXXX.pt
--sample-index 0 --out bev_sample.png` (checkpoint/real-data path not
exercised by this repo's test suite; the plotting function itself is).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_bev_predictions(
    seg_logits: torch.Tensor,
    det_heatmap: torch.Tensor,
    det_offset: torch.Tensor,
    out_path: str | Path,
    class_names: tuple[str, ...] = ("drivable_area", "lane"),
    score_threshold: float = 0.3,
    title: str | None = None,
) -> Path:
    """
    Args:
        seg_logits: (num_seg_classes, H, W) raw logits.
        det_heatmap: (num_det_classes, H, W) raw logits.
        det_offset: (2, H, W) sub-cell offset predictions.
        out_path: file to save the figure to (parent dirs created as needed).

    Returns the resolved `out_path`.
    """
    seg_probs = torch.sigmoid(seg_logits).detach().cpu().numpy()
    det_scores = torch.sigmoid(det_heatmap).detach().cpu().numpy()
    offset = det_offset.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 6))

    overlay = np.zeros((*seg_probs.shape[1:], 3), dtype=np.float32)
    colors = plt.cm.tab10(np.linspace(0, 1, max(seg_probs.shape[0], 1)))
    for class_idx in range(seg_probs.shape[0]):
        mask = seg_probs[class_idx] > 0.5
        overlay[mask] = colors[class_idx][:3]
    ax.imshow(overlay, origin="lower")

    for class_idx in range(det_scores.shape[0]):
        ys, xs = np.nonzero(det_scores[class_idx] > score_threshold)
        for y, x in zip(ys, xs):
            px = x + offset[0, y, x]
            py = y + offset[1, y, x]
            ax.plot(px, py, marker="x", markersize=8, color="red")
            ax.text(px, py, class_names[class_idx] if class_idx < len(class_names) else str(class_idx), fontsize=6, color="white")

    ax.set_title(title or "BEV predictions")
    ax.set_xlabel("x (grid cells)")
    ax.set_ylabel("y (grid cells)")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--out", type=str, default="bev_sample.png")
    return parser.parse_args()


def main() -> None:
    raise NotImplementedError(
        "Loading a real checkpoint + nuScenes-mini sample requires the downloaded dataset, "
        "which is not present in this environment. Call plot_bev_predictions(...) directly "
        "with model outputs once both are available; see tests/test_visualize_bev.py for an "
        "example invocation on synthetic data."
    )


if __name__ == "__main__":
    _parse_args()
    main()
