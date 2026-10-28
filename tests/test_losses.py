import pytest
import torch

from roadvision.losses import DepthLoss, DetectionLoss, MultiTaskUncertaintyLoss, SegmentationLoss
from roadvision.losses.detection_loss import gaussian_focal_loss, masked_l1_loss


class TestDepthLoss:
    def _inputs(self):
        torch.manual_seed(0)
        dbound = (4.0, 12.0, 4.0)  # 2 bins: centered at 4, 8
        num_bins, stride = 2, 2
        logits = torch.randn(1, 1, num_bins, 2, 2, requires_grad=True)
        depth_dist = logits.softmax(dim=2)
        depth_maps = torch.zeros(1, 1, 4, 4)
        depth_maps[0, 0, 0, 0] = 4.0  # one valid LIDAR return, bin 0
        return dbound, stride, depth_dist, depth_maps, logits

    def test_finite_and_backward(self):
        dbound, stride, depth_dist, depth_maps, logits = self._inputs()
        loss_fn = DepthLoss(dbound, backbone_downsample=stride)
        loss = loss_fn(depth_dist, depth_maps)
        assert torch.isfinite(loss)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_no_valid_pixels_is_zero(self):
        dbound, stride, depth_dist, _, logits = self._inputs()
        depth_maps = torch.zeros(1, 1, 4, 4)
        loss_fn = DepthLoss(dbound, backbone_downsample=stride)
        loss = loss_fn(depth_dist, depth_maps)
        assert loss.item() == 0.0

    def test_masked_block_does_not_affect_loss(self):
        dbound, stride, depth_dist, depth_maps, _ = self._inputs()
        loss_fn = DepthLoss(dbound, backbone_downsample=stride)
        baseline = loss_fn(depth_dist.detach().clone(), depth_maps)

        perturbed = depth_dist.detach().clone()
        perturbed[0, 0, :, 1, 1] = torch.tensor([0.99, 0.01])  # block (1,1) has no LIDAR return
        perturbed_loss = loss_fn(perturbed, depth_maps)
        assert torch.isclose(baseline, perturbed_loss)


class TestSegmentationLoss:
    def test_finite_and_backward(self):
        logits = torch.randn(2, 2, 8, 8, requires_grad=True)
        targets = (torch.rand(2, 2, 8, 8) > 0.7).float()
        loss = SegmentationLoss()(logits, targets)
        assert torch.isfinite(loss)
        loss.backward()
        assert logits.grad is not None

    def test_confident_correct_prediction_has_lower_loss(self):
        targets = torch.zeros(1, 1, 4, 4)
        targets[0, 0, 0, 0] = 1.0
        loss_fn = SegmentationLoss()

        confident_correct = torch.full((1, 1, 4, 4), -10.0)
        confident_correct[0, 0, 0, 0] = 10.0
        confidently_wrong = torch.full((1, 1, 4, 4), 10.0)
        confidently_wrong[0, 0, 0, 0] = -10.0

        assert loss_fn(confident_correct, targets) < loss_fn(confidently_wrong, targets)


class TestDetectionLoss:
    def _targets(self, h=8, w=8):
        heatmap = torch.zeros(1, 3, h, w)
        heatmap[0, 0, 4, 4] = 1.0
        mask = torch.zeros(1, h, w)
        mask[0, 4, 4] = 1.0
        return {
            "heatmap": heatmap,
            "offset": torch.zeros(1, 2, h, w),
            "dims": torch.zeros(1, 2, h, w),
            "yaw": torch.zeros(1, 2, h, w),
            "velocity": torch.zeros(1, 2, h, w),
            "mask": mask,
        }

    def _pred(self, h=8, w=8, requires_grad=True):
        return {
            key: torch.zeros(1, 3 if key == "heatmap" else 2, h, w, requires_grad=requires_grad)
            for key in ("heatmap", "offset", "dims", "yaw", "velocity")
        }

    def test_finite_and_backward(self):
        pred = self._pred()
        target = self._targets()
        loss = DetectionLoss()(pred, target)
        assert torch.isfinite(loss)
        loss.backward()
        for tensor in pred.values():
            assert tensor.grad is not None

    def test_regression_ignores_non_center_cells(self):
        target = self._targets()
        pred_a = self._pred(requires_grad=False)
        pred_b = {k: v.clone() for k, v in pred_a.items()}
        pred_b["offset"][0, :, 0, 0] = 5.0  # (0, 0) is not the object center

        loss_a = DetectionLoss()(pred_a, target)
        loss_b = DetectionLoss()(pred_b, target)
        assert torch.isclose(loss_a, loss_b)

    def test_gaussian_focal_loss_penalizes_false_positive_more_than_near_miss(self):
        target = torch.zeros(1, 1, 5, 5)
        target[0, 0, 2, 2] = 1.0
        target[0, 0, 2, 3] = 0.8  # Gaussian tail next to the true center

        logits_far_false_positive = torch.full((1, 1, 5, 5), -10.0)
        logits_far_false_positive[0, 0, 4, 4] = 10.0
        logits_near_miss = torch.full((1, 1, 5, 5), -10.0)
        logits_near_miss[0, 0, 2, 3] = 10.0

        far_loss = gaussian_focal_loss(logits_far_false_positive, target)
        near_loss = gaussian_focal_loss(logits_near_miss, target)
        assert near_loss < far_loss

    def test_masked_l1_loss_zero_when_mask_empty(self):
        pred = torch.randn(1, 2, 4, 4)
        target = torch.randn(1, 2, 4, 4)
        mask = torch.zeros(1, 4, 4)
        assert masked_l1_loss(pred, target, mask).item() == 0.0

    def test_per_class_normalize_gives_rare_class_comparable_gradient(self):
        """Regression test for a real Phase 5 finding: nuScenes-mini's real
        class counts are ~7600 `car` vs ~60 `trailer`. Global normalization
        (dividing total loss by the total center count pooled over all
        classes) lets a common class's gradient swamp a rare class's. With
        one "common" class carrying 20 centers and one "rare" class carrying
        1, per-class normalization must give the rare class's single center
        a gradient on the same order as the common class's centers -- not
        ~20x smaller.
        """
        h = w = 16
        target = torch.zeros(1, 2, h, w)
        target[0, 0].view(-1)[:20] = 1.0  # "common" class: 20 centers
        target[0, 1, 0, 0] = 1.0  # "rare" class: 1 center

        logits = torch.zeros(1, 2, h, w, requires_grad=True)
        loss = gaussian_focal_loss(logits, target, per_class_normalize=True)
        loss.backward()

        common_grad = logits.grad[0, 0].view(-1)[:20].abs().mean()
        rare_grad = logits.grad[0, 1, 0, 0].abs()
        assert rare_grad / common_grad > 0.5  # same order of magnitude, not ~20x smaller

    def test_per_class_normalize_differs_from_global(self):
        h = w = 16
        target = torch.zeros(1, 2, h, w)
        target[0, 0].view(-1)[:20] = 1.0
        target[0, 1, 0, 0] = 1.0
        logits = torch.randn(1, 2, h, w)

        global_loss = gaussian_focal_loss(logits, target, per_class_normalize=False)
        per_class_loss = gaussian_focal_loss(logits, target, per_class_normalize=True)
        assert not torch.isclose(global_loss, per_class_loss)


class TestMultiTaskUncertaintyLoss:
    def test_finite_and_gradients_flow_to_log_vars_and_task_losses(self):
        loss_module = MultiTaskUncertaintyLoss()
        x = torch.tensor(2.0, requires_grad=True)
        task_losses = {"depth": x**2, "segmentation": x.abs(), "detection": x * 3}

        out = loss_module(task_losses)
        assert torch.isfinite(out["total"])
        out["total"].backward()

        assert loss_module.log_vars.grad is not None
        assert torch.isfinite(loss_module.log_vars.grad).all()
        assert x.grad is not None

    def test_missing_task_raises(self):
        loss_module = MultiTaskUncertaintyLoss()
        with pytest.raises(ValueError):
            loss_module({"depth": torch.tensor(1.0), "segmentation": torch.tensor(1.0)})
