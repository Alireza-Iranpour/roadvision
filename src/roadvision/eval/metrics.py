"""Per-task evaluation metrics.

Depth and segmentation metrics are standard, exact definitions (AbsRel/
RMSE/delta-thresholds; per-class IoU). The detection metric is an explicitly
*simplified proxy* for nuScenes' official mAP/NDS: center-distance matching
plus PR-curve AP, computed per class at a few distance thresholds, without
the official eval's full attribute/velocity error terms or its exact
distance-threshold set. It is useful for tracking relative progress and
catching regressions during development, not for reporting a number
comparable to the nuScenes detection leaderboard -- doing that would
require depth-matching against `nuscenes-devkit`'s own `DetectionEval`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from roadvision.data.geometry import make_depth_bins


@dataclass(frozen=True)
class DepthMetrics:
    abs_rel: float
    sq_rel: float
    rmse: float
    rmse_log: float
    delta1: float  # fraction of pixels with max(pred/gt, gt/pred) < 1.25
    delta2: float  # ... < 1.25^2
    delta3: float  # ... < 1.25^3


def expected_depth_from_distribution(depth_dist: torch.Tensor, dbound: tuple[float, float, float]) -> torch.Tensor:
    """depth_dist: (..., D, H, W) per-bin probabilities (sums to 1 over D).

    Returns the probability-weighted mean depth (..., H, W), not the argmax
    bin: the expectation is differentiable-equivalent to what the model was
    trained against and avoids discretization error from a single bin
    center, at the cost of being pulled toward the mean on multi-modal
    distributions (e.g. depth discontinuities at object silhouettes).
    """
    bin_centers = torch.from_numpy(make_depth_bins(dbound)).to(depth_dist.device, depth_dist.dtype)
    num_bins = depth_dist.shape[-3]
    if bin_centers.shape[0] != num_bins:
        raise ValueError(f"dbound {dbound} produces {bin_centers.shape[0]} bins, depth_dist has {num_bins}")
    shape = (1,) * (depth_dist.dim() - 3) + (num_bins, 1, 1)
    return (depth_dist * bin_centers.view(shape)).sum(dim=-3)


def compute_depth_metrics(pred_depth: torch.Tensor, gt_depth: torch.Tensor, valid_mask: torch.Tensor) -> DepthMetrics:
    """pred_depth, gt_depth, valid_mask: same shape, e.g. (B, N, H, W).
    `valid_mask` marks pixels with a real LIDAR-projected depth GT value
    (see `losses/depth_loss.py`'s `_downsample_sparse_depth`) -- metrics are
    computed over valid pixels only.
    """
    if not valid_mask.any():
        nan = float("nan")
        return DepthMetrics(nan, nan, nan, nan, nan, nan, nan)

    pred = pred_depth[valid_mask].double().clamp_min(1e-6)
    gt = gt_depth[valid_mask].double().clamp_min(1e-6)

    diff = pred - gt
    abs_rel = (diff.abs() / gt).mean().item()
    sq_rel = (diff.pow(2) / gt).mean().item()
    rmse = diff.pow(2).mean().sqrt().item()
    rmse_log = (pred.log() - gt.log()).pow(2).mean().sqrt().item()

    ratio = torch.maximum(pred / gt, gt / pred)
    delta1 = (ratio < 1.25).double().mean().item()
    delta2 = (ratio < 1.25**2).double().mean().item()
    delta3 = (ratio < 1.25**3).double().mean().item()

    return DepthMetrics(abs_rel, sq_rel, rmse, rmse_log, delta1, delta2, delta3)


def compute_segmentation_iou(pred_logits: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    """pred_logits, gt_mask: (B, C, H, W) / (B, C, H, W) multi-label (per
    `models/heads/segmentation_head.py`: classes are independent, not a
    softmax partition). Returns per-class IoU plus "mean_iou", where a class
    absent from both prediction and target in the whole batch contributes
    IoU=1.0 (perfect agreement on "nothing here"), matching the convention
    `tests/test_metrics.py` asserts against.
    """
    pred = (torch.sigmoid(pred_logits) > threshold).bool()
    gt = gt_mask.bool()

    num_classes = pred.shape[1]
    ious: dict[str, float] = {}
    for c in range(num_classes):
        intersection = (pred[:, c] & gt[:, c]).sum().double()
        union = (pred[:, c] | gt[:, c]).sum().double()
        iou = 1.0 if union == 0 else (intersection / union).item()
        ious[f"class_{c}_iou"] = iou

    ious["mean_iou"] = sum(ious.values()) / num_classes
    return ious


@dataclass(frozen=True)
class DetectionPRResult:
    precision: float
    recall: float
    ap: float
    num_gt: int
    num_pred: int
    num_tp: int


def _heatmap_peaks(heatmap: torch.Tensor, score_threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """heatmap: (H, W) sigmoid scores for one class in one sample. Returns
    (scores, (y, x) indices) of local 3x3-maxima above `score_threshold`,
    the usual CenterNet decoding (NMS-by-local-maximum instead of a
    separate NMS pass since targets are single-peak Gaussians per object).
    """
    pooled = torch.nn.functional.max_pool2d(heatmap[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    is_peak = (pooled == heatmap) & (heatmap > score_threshold)
    ys, xs = torch.nonzero(is_peak, as_tuple=True)
    return heatmap[ys, xs], torch.stack([ys, xs], dim=-1)


def _cell_to_xy(indices: torch.Tensor, offset: torch.Tensor, grid_xres: float, grid_yres: float) -> torch.Tensor:
    """indices: (N, 2) (y, x) cell indices. offset: (2, H, W) sub-cell
    offset predictions (see `bev_rasterize.DetectionTargets.offset`).
    Returns (N, 2) (x, y) positions in grid-cell units (not meters) -- the
    distance threshold in `match_detections_by_center_distance` is likewise
    expressed in grid cells, so resolution cancels out; callers passing a
    metric threshold must scale by (grid_xres, grid_yres) themselves.
    """
    off = offset[:, indices[:, 0], indices[:, 1]].T  # (N, 2) = (dx, dy)
    return torch.stack([indices[:, 1].double() + off[:, 0], indices[:, 0].double() + off[:, 1]], dim=-1)


def match_detections_by_center_distance(
    pred_heatmap: torch.Tensor,
    pred_offset: torch.Tensor,
    gt_centers_xy: torch.Tensor,
    distance_threshold: float,
    score_threshold: float = 0.1,
) -> DetectionPRResult:
    """Single-sample, single-class center-distance matching (nuScenes-style:
    match by 2D center distance in BEV, not by box IoU, since small/thin BEV
    boxes make IoU an unstable signal). Each GT is matched to at most one
    prediction (highest score first, greedy), each prediction to at most one
    GT.

    pred_heatmap: (H, W). pred_offset: (2, H, W). gt_centers_xy: (num_gt, 2)
    cell-unit coordinates (consistent with `_cell_to_xy`'s output). Returns
    precision/recall at `score_threshold` plus a continuous-recall AP
    (area under the precision-recall curve as score threshold varies),
    which is more informative than a single-threshold PR pair alone.
    """
    scores, indices = _heatmap_peaks(pred_heatmap, score_threshold=1e-4)
    order = torch.argsort(scores, descending=True)
    scores, indices = scores[order], indices[order]
    pred_xy = _cell_to_xy(indices, pred_offset, 1.0, 1.0) if len(indices) else torch.zeros((0, 2))

    num_gt = gt_centers_xy.shape[0]
    gt_matched = torch.zeros(num_gt, dtype=torch.bool)
    tp = torch.zeros(len(scores), dtype=torch.bool)

    for i in range(len(scores)):
        if num_gt == 0:
            break
        dists = (gt_centers_xy - pred_xy[i]).pow(2).sum(dim=-1).sqrt()
        dists = dists.masked_fill(gt_matched, float("inf"))
        best = torch.argmin(dists) if num_gt > 0 else None
        if best is not None and dists[best] < distance_threshold:
            tp[i] = True
            gt_matched[best] = True

    cum_tp = torch.cumsum(tp.double(), dim=0)
    cum_fp = torch.cumsum((~tp).double(), dim=0)
    precisions = cum_tp / (cum_tp + cum_fp).clamp_min(1)
    recalls = cum_tp / max(num_gt, 1)

    # Continuous-recall AP: integrate precision over recall (trapezoidal),
    # with a precision-envelope max applied first (standard PR-AP practice)
    # so a later better-precision point isn't suppressed by an earlier dip.
    if len(scores) == 0 or num_gt == 0:
        ap = 0.0 if num_gt > 0 else 1.0
    else:
        precisions_env = torch.flip(torch.cummax(torch.flip(precisions, dims=[0]), dim=0).values, dims=[0])
        recalls_padded = torch.cat([torch.zeros(1), recalls])
        precisions_padded = torch.cat([precisions_env[:1], precisions_env])
        ap = torch.trapz(precisions_padded, recalls_padded).item()

    at_threshold = scores >= score_threshold
    tp_at_threshold = int(tp[at_threshold].sum().item())
    pred_at_threshold = int(at_threshold.sum().item())

    return DetectionPRResult(
        precision=tp_at_threshold / pred_at_threshold if pred_at_threshold else 0.0,
        recall=tp_at_threshold / num_gt if num_gt else 1.0,
        ap=ap,
        num_gt=num_gt,
        num_pred=pred_at_threshold,
        num_tp=tp_at_threshold,
    )
