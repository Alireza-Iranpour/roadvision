import numpy as np
import torch

from roadvision.data.geometry import (
    BEVGrid,
    camera_to_bev_indices,
    generate_frustum,
    transform_points,
)
from roadvision.models.bev_pool import (
    BEVPool,
    FrustumConfig,
    generate_frustum_torch,
    points_to_bev_indices_torch,
    transform_points_torch,
)

_INTRINSIC = np.array([[50.0, 0.0, 50.0], [0.0, 50.0, 50.0], [0.0, 0.0, 1.0]])
_DEPTH_BINS = np.array([5.0, 10.0, 15.0])
_GRID = BEVGrid(xbound=(-10.0, 10.0, 1.0), ybound=(-10.0, 10.0, 1.0))


def _cam_to_ego_rotated() -> np.ndarray:
    # Same 90deg-about-z rotation used in test_geometry.py's cam_to_ego test.
    return np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


class TestGenerateFrustumTorchMatchesNumpy:
    def test_matches_reference(self):
        expected = generate_frustum(
            image_size=(100, 100), feature_size=(10, 10), depth_bins=_DEPTH_BINS, intrinsic=_INTRINSIC
        )
        intrinsic_t = torch.from_numpy(_INTRINSIC).float().view(1, 1, 3, 3)
        depth_bins_t = torch.from_numpy(_DEPTH_BINS).float()
        config = FrustumConfig(image_size=(100, 100), feature_size=(10, 10))

        actual = generate_frustum_torch(config, depth_bins_t, intrinsic_t)
        assert actual.shape == (1, 1, 3, 10, 10, 3)
        np.testing.assert_allclose(actual[0, 0].numpy(), expected, rtol=1e-5, atol=1e-5)


class TestTransformPointsTorchMatchesNumpy:
    def test_matches_reference(self):
        points_cam = np.random.default_rng(0).normal(size=(3, 4, 3))
        transform = _cam_to_ego_rotated()

        expected = transform_points(points_cam, transform)
        actual = transform_points_torch(
            torch.from_numpy(points_cam).float().view(1, 1, 3, 4, 3),
            torch.from_numpy(transform).float().view(1, 1, 4, 4),
        )
        np.testing.assert_allclose(actual.view(3, 4, 3).numpy(), expected, rtol=1e-5, atol=1e-5)


class TestPointsToBevIndicesTorchMatchesNumpy:
    def test_matches_reference(self):
        points_cam = np.array([[5.0, 0.0, 1.5], [100.0, 0.0, 1.5]])
        transform = np.eye(4)

        expected_indices, expected_valid = camera_to_bev_indices(points_cam, transform, _GRID)
        points_ego = torch.from_numpy(transform_points(points_cam, transform)).float()
        actual_indices, actual_valid = points_to_bev_indices_torch(points_ego, _GRID)

        np.testing.assert_array_equal(actual_indices.numpy(), expected_indices)
        np.testing.assert_array_equal(actual_valid.numpy(), expected_valid)


class TestBEVPoolSingleKnownPoint:
    """A single frustum point with all its feature mass should land in
    exactly the BEV grid cell predicted by the (already-tested) pure
    geometry functions -- this is the test that would catch an LSS-style
    axis/convention mismatch between the lift and splat steps."""

    def test_point_lands_in_expected_cell(self):
        feat_h, feat_w = 4, 4
        target_row, target_col = 1, 2

        intrinsic = np.array([[8.0, 0.0, 16.0], [0.0, 8.0, 16.0], [0.0, 0.0, 1.0]])
        depth_bins = np.array([10.0])
        cam_to_ego = np.eye(4)

        frustum = generate_frustum(
            image_size=(32, 32), feature_size=(feat_h, feat_w), depth_bins=depth_bins, intrinsic=intrinsic
        )
        expected_indices, expected_valid = camera_to_bev_indices(
            frustum[0, target_row, target_col][None, :], cam_to_ego, _GRID
        )
        assert expected_valid[0]
        expected_ix, expected_iy = expected_indices[0]

        config = FrustumConfig(image_size=(32, 32), feature_size=(feat_h, feat_w))
        pool = BEVPool(config, torch.from_numpy(depth_bins).float(), _GRID)

        depth_dist = torch.ones(1, 1, 1, feat_h, feat_w)  # D=1 -> trivially normalized
        context = torch.zeros(1, 1, 1, feat_h, feat_w)
        context[0, 0, 0, target_row, target_col] = 1.0

        bev = pool(
            depth_dist,
            context,
            torch.from_numpy(intrinsic).float().view(1, 1, 3, 3),
            torch.from_numpy(cam_to_ego).float().view(1, 1, 4, 4),
        )

        assert bev.shape == (1, 1, _GRID.ny, _GRID.nx)
        assert bev[0, 0, expected_iy, expected_ix].item() == 1.0
        assert bev.sum().item() == 1.0

    def test_out_of_grid_point_contributes_nothing(self):
        feat_h, feat_w = 2, 2
        # cx=cy=0 so every ray has a nonzero (x, y) slope (no pixel maps to
        # the degenerate straight-ahead (0, 0, depth) ray that would stay
        # inside the grid at any depth).
        intrinsic = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        depth_bins = np.array([1000.0])  # far enough to land outside _GRID
        cam_to_ego = np.eye(4)

        config = FrustumConfig(image_size=(4, 4), feature_size=(feat_h, feat_w))
        pool = BEVPool(config, torch.from_numpy(depth_bins).float(), _GRID)

        depth_dist = torch.ones(1, 1, 1, feat_h, feat_w)
        context = torch.ones(1, 1, 1, feat_h, feat_w)

        bev = pool(
            depth_dist,
            context,
            torch.from_numpy(intrinsic).float().view(1, 1, 3, 3),
            torch.from_numpy(cam_to_ego).float().view(1, 1, 4, 4),
        )
        assert bev.sum().item() == 0.0
