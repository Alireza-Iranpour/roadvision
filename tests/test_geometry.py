import numpy as np
import pytest

from roadvision.data.geometry import (
    BEVGrid,
    camera_to_bev_indices,
    generate_frustum,
    make_depth_bins,
    project_lidar_to_camera,
    transform_points,
)


def _identity() -> np.ndarray:
    return np.eye(4)


def _translation(tx: float, ty: float, tz: float) -> np.ndarray:
    t = np.eye(4)
    t[:3, 3] = [tx, ty, tz]
    return t


class TestTransformPoints:
    def test_identity_is_noop(self):
        points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 5.0]])
        out = transform_points(points, _identity())
        np.testing.assert_allclose(out, points)

    def test_translation(self):
        points = np.array([[0.0, 0.0, 0.0]])
        out = transform_points(points, _translation(1.0, -2.0, 3.0))
        np.testing.assert_allclose(out, [[1.0, -2.0, 3.0]])

    def test_preserves_leading_shape(self):
        points = np.zeros((2, 3, 3))
        out = transform_points(points, _identity())
        assert out.shape == (2, 3, 3)


class TestProjectLidarToCamera:
    def test_known_pinhole_projection(self):
        # fx=fy=100, cx=cy=50; point at camera-frame (1, 2, 10).
        intrinsic = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
        points_ego = np.array([[1.0, 2.0, 10.0]])
        pixels, depths, valid = project_lidar_to_camera(points_ego, intrinsic, _identity())

        np.testing.assert_allclose(pixels, [[60.0, 70.0]])
        np.testing.assert_allclose(depths, [10.0])
        assert valid[0]

    def test_point_behind_camera_is_invalid(self):
        intrinsic = np.eye(3)
        points_ego = np.array([[1.0, 1.0, -5.0]])
        _, depths, valid = project_lidar_to_camera(points_ego, intrinsic, _identity())

        assert depths[0] == pytest.approx(-5.0)
        assert not valid[0]

    def test_cam_to_ego_offset_shifts_projection(self):
        # Camera sits 5m forward (ego +x) of ego origin, facing same way.
        intrinsic = np.eye(3)
        cam_to_ego = _translation(5.0, 0.0, 0.0)
        # A point 5m ahead of the camera in camera frame is 10m ahead in ego frame.
        point_ego = np.array([[10.0, 0.0, 5.0]])
        _, depths, valid = project_lidar_to_camera(point_ego, intrinsic, cam_to_ego)

        np.testing.assert_allclose(depths, [5.0])
        assert valid[0]


class TestMakeDepthBins:
    def test_known_dbound(self):
        bins = make_depth_bins((4.0, 45.0, 1.0))
        assert bins.shape == (41,)
        np.testing.assert_allclose(bins[0], 4.0)
        np.testing.assert_allclose(bins[-1], 44.0)

    def test_step_spacing(self):
        bins = make_depth_bins((5.0, 10.0, 2.0))
        np.testing.assert_allclose(bins, [5.0, 7.0, 9.0])


class TestGenerateFrustum:
    def test_shape_and_depth_exactness(self):
        depth_bins = np.array([5.0, 10.0, 15.0])
        intrinsic = np.array([[50.0, 0.0, 50.0], [0.0, 50.0, 50.0], [0.0, 0.0, 1.0]])
        frustum = generate_frustum(
            image_size=(100, 100), feature_size=(10, 10), depth_bins=depth_bins, intrinsic=intrinsic
        )

        assert frustum.shape == (3, 10, 10, 3)
        # z-coordinate of every point must equal its depth bin exactly,
        # since the frustum is built by scaling a unit-depth ray.
        for d_idx, d in enumerate(depth_bins):
            np.testing.assert_allclose(frustum[d_idx, :, :, 2], d)

    def test_known_ray_unprojection(self):
        intrinsic = np.array([[50.0, 0.0, 50.0], [0.0, 50.0, 50.0], [0.0, 0.0, 1.0]])
        frustum = generate_frustum(
            image_size=(100, 100), feature_size=(10, 10), depth_bins=np.array([5.0]), intrinsic=intrinsic
        )
        # First feature cell center: u = v = (0 + 0.5) * (100 / 10) = 5.
        # ray = inv(K) @ [5, 5, 1] = [(5-50)/50, (5-50)/50, 1] = [-0.9, -0.9, 1]
        # point at depth 5 = [-4.5, -4.5, 5]
        np.testing.assert_allclose(frustum[0, 0, 0], [-4.5, -4.5, 5.0])


class TestCameraToBevIndices:
    def test_known_grid_cell(self):
        grid = BEVGrid(xbound=(-50.0, 50.0, 0.5), ybound=(-50.0, 50.0, 0.5))
        points_cam = np.array([[5.0, 0.0, 1.5]])
        indices, valid = camera_to_bev_indices(points_cam, _identity(), grid)

        # ix = floor((5 - (-50)) / 0.5) = floor(110) = 110
        # iy = floor((0 - (-50)) / 0.5) = floor(100) = 100
        np.testing.assert_array_equal(indices, [[110, 100]])
        assert valid[0]

    def test_out_of_grid_point_is_invalid(self):
        grid = BEVGrid(xbound=(-50.0, 50.0, 0.5), ybound=(-50.0, 50.0, 0.5))
        points_cam = np.array([[100.0, 0.0, 1.5]])
        _, valid = camera_to_bev_indices(points_cam, _identity(), grid)
        assert not valid[0]

    def test_grid_origin_cell(self):
        grid = BEVGrid(xbound=(-50.0, 50.0, 0.5), ybound=(-50.0, 50.0, 0.5))
        points_cam = np.array([[-50.0, -50.0, 1.0]])
        indices, valid = camera_to_bev_indices(points_cam, _identity(), grid)
        np.testing.assert_array_equal(indices, [[0, 0]])
        assert valid[0]

    def test_cam_to_ego_rotation_is_applied(self):
        # Camera frame rotated 90deg about ego z-axis: cam +x axis points
        # along ego +y. A point at camera-frame (10, 0, 0) should land at
        # ego (0, 10).
        grid = BEVGrid(xbound=(-50.0, 50.0, 1.0), ybound=(-50.0, 50.0, 1.0))
        cam_to_ego = np.array(
            [
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        points_cam = np.array([[10.0, 0.0, 0.0]])
        indices, valid = camera_to_bev_indices(points_cam, cam_to_ego, grid)
        np.testing.assert_array_equal(indices, [[50, 60]])
        assert valid[0]
