import math

import torch

from roadvision.eval.metrics import (
    compute_depth_metrics,
    compute_segmentation_iou,
    expected_depth_from_distribution,
    match_detections_by_center_distance,
)


def test_expected_depth_from_distribution_onehot():
    dbound = (4.0, 45.0, 1.0)
    bin_index = 3  # center = 4.0 + 3 * 1.0 = 7.0
    depth_dist = torch.zeros(1, 1, 41, 2, 2)
    depth_dist[:, :, bin_index] = 1.0

    expected = expected_depth_from_distribution(depth_dist, dbound)

    assert expected.shape == (1, 1, 2, 2)
    assert torch.allclose(expected, torch.full((1, 1, 2, 2), 7.0), atol=1e-5)


def test_depth_metrics_known_offset():
    gt = torch.tensor([10.0, 20.0])
    pred = torch.tensor([11.0, 999.0])  # second entry is a huge outlier
    mask = torch.tensor([True, False])  # ...but it's masked out

    metrics = compute_depth_metrics(pred, gt, mask)

    assert math.isclose(metrics.abs_rel, 0.1, rel_tol=1e-6)
    assert math.isclose(metrics.sq_rel, 0.1, rel_tol=1e-6)
    assert math.isclose(metrics.rmse, 1.0, rel_tol=1e-6)
    assert math.isclose(metrics.rmse_log, math.log(1.1), rel_tol=1e-5)
    assert metrics.delta1 == 1.0
    assert metrics.delta2 == 1.0
    assert metrics.delta3 == 1.0


def test_depth_metrics_all_invalid_returns_nan():
    gt = torch.tensor([10.0, 20.0])
    pred = torch.tensor([11.0, 21.0])
    mask = torch.zeros(2, dtype=torch.bool)

    metrics = compute_depth_metrics(pred, gt, mask)

    assert math.isnan(metrics.abs_rel)


def test_segmentation_iou_perfect_absent_and_wrong_classes():
    # class 0: perfect match -> IoU 1.0
    # class 1: prediction and GT both empty -> IoU 1.0 (vacuous agreement)
    # class 2: prediction everywhere, GT nowhere -> IoU 0.0
    pred_logits = torch.stack(
        [
            torch.full((2, 2), 10.0),
            torch.full((2, 2), -10.0),
            torch.full((2, 2), 10.0),
        ]
    ).unsqueeze(0)
    gt_mask = torch.stack(
        [
            torch.ones(2, 2),
            torch.zeros(2, 2),
            torch.zeros(2, 2),
        ]
    ).unsqueeze(0)

    ious = compute_segmentation_iou(pred_logits, gt_mask)

    assert math.isclose(ious["class_0_iou"], 1.0)
    assert math.isclose(ious["class_1_iou"], 1.0)
    assert math.isclose(ious["class_2_iou"], 0.0)
    assert math.isclose(ious["mean_iou"], 2.0 / 3.0, rel_tol=1e-6)


def test_detection_matching_true_positive_and_false_positive():
    heatmap = torch.zeros(5, 5)
    heatmap[2, 2] = 0.9
    heatmap[0, 0] = 0.3

    offset = torch.zeros(2, 5, 5)
    offset[:, 2, 2] = torch.tensor([0.3, 0.0])  # (dx, dy) -> cell-space (2.3, 2.0)

    gt_centers_xy = torch.tensor([[2.3, 2.0]])

    result = match_detections_by_center_distance(
        heatmap, offset, gt_centers_xy, distance_threshold=0.5, score_threshold=0.1
    )

    assert result.num_gt == 1
    assert result.num_pred == 2
    assert result.num_tp == 1
    assert math.isclose(result.precision, 0.5)
    assert math.isclose(result.recall, 1.0)
    assert math.isclose(result.ap, 1.0, rel_tol=1e-5)


def test_detection_matching_no_predictions_no_gt_is_vacuously_perfect():
    heatmap = torch.zeros(5, 5)
    offset = torch.zeros(2, 5, 5)
    gt_centers_xy = torch.zeros(0, 2)

    result = match_detections_by_center_distance(heatmap, offset, gt_centers_xy, distance_threshold=0.5)

    assert result.num_gt == 0
    assert result.num_pred == 0
    assert result.recall == 1.0
    assert result.ap == 1.0


def test_detection_matching_unmatched_gt_scores_zero():
    heatmap = torch.zeros(5, 5)
    heatmap[0, 0] = 0.3  # far from the only GT, so it's a false positive

    offset = torch.zeros(2, 5, 5)
    gt_centers_xy = torch.tensor([[4.0, 4.0]])

    result = match_detections_by_center_distance(heatmap, offset, gt_centers_xy, distance_threshold=0.5)

    assert result.num_tp == 0
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.ap == 0.0


def test_gt_centers_for_class_isolates_per_class_centers():
    """Regression test for a real Phase 5 bug: `evaluate_dataset` originally
    used `det_target["mask"]` (every class's object centers, one shared
    channel) as ground truth for EVERY class's AP, so e.g. a `truck`'s true
    position could count as a correct match for the `car` head. Ground
    truth for class c must come only from that class's own heatmap channel.
    """
    from roadvision.eval.evaluate import gt_centers_for_class

    num_classes, h, w = 2, 5, 5
    heatmap = torch.zeros(1, num_classes, h, w)
    heatmap[0, 0, 1, 1] = 1.0  # class 0 object at (1, 1)
    heatmap[0, 1, 3, 3] = 1.0  # class 1 object at (3, 3)
    det_target = {"heatmap": heatmap, "offset": torch.zeros(1, 2, h, w)}

    class_0_centers = gt_centers_for_class(det_target, batch_idx=0, class_idx=0)
    class_1_centers = gt_centers_for_class(det_target, batch_idx=0, class_idx=1)

    assert class_0_centers.shape == (1, 2)
    assert torch.allclose(class_0_centers[0], torch.tensor([1.0, 1.0], dtype=torch.float64))
    assert class_1_centers.shape == (1, 2)
    assert torch.allclose(class_1_centers[0], torch.tensor([3.0, 3.0], dtype=torch.float64))
