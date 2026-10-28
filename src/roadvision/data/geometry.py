"""Pure camera/BEV projection geometry.

Conventions used throughout this module and the rest of roadvision:
  - Camera frame: OpenCV/pinhole convention, +x right, +y down, +z forward
    (out of the lens). A point's z-coordinate in camera frame is its depth.
  - Ego frame: +x forward, +y left, +z up (ISO 8855, matches nuScenes).
  - BEV grid: 2D grid over the ego frame's (x, y) plane. Grid index (0, 0)
    is the cell at (xbound.min, ybound.min); index increases with x then y.
  - All 4x4 transforms are homogeneous, right-multiplying a column point:
    p_dst = T @ [p_src; 1]. A "cam_to_ego" transform maps a point expressed
    in camera frame to the same point expressed in ego frame.

No function here touches disk, the network, or any third-party dataset
API -- that keeps this module exactly unit-testable from known inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BEVGrid:
    """Defines a 2D BEV grid over the ego-frame (x, y) plane."""

    xbound: tuple[float, float, float]  # (min, max, resolution), meters
    ybound: tuple[float, float, float]

    @property
    def nx(self) -> int:
        xmin, xmax, xres = self.xbound
        return int(round((xmax - xmin) / xres))

    @property
    def ny(self) -> int:
        ymin, ymax, yres = self.ybound
        return int(round((ymax - ymin) / yres))


def make_depth_bins(dbound: tuple[float, float, float]) -> np.ndarray:
    """Build the discretized depth-bin centers used by the depth head and
    the lift-splat frustum, from a (min, max, step) bound."""
    min_depth, max_depth, step = dbound
    return np.arange(min_depth, max_depth, step, dtype=np.float64)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to an (..., 3) array of points."""
    if transform.shape != (4, 4):
        raise ValueError(f"expected a 4x4 transform, got shape {transform.shape}")
    orig_shape = points.shape
    flat = points.reshape(-1, 3)
    ones = np.ones((flat.shape[0], 1), dtype=flat.dtype)
    homogeneous = np.concatenate([flat, ones], axis=1)
    transformed = homogeneous @ transform.T
    return transformed[:, :3].reshape(orig_shape)


def project_lidar_to_camera(
    points_ego: np.ndarray,
    cam_intrinsic: np.ndarray,
    cam_to_ego: np.ndarray,
    min_depth: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project points expressed in ego frame into a camera's pixel plane.

    `points_ego` must already be expressed in the ego frame at the camera's
    own timestamp (i.e. any lidar-timestamp -> global -> camera-timestamp
    ego-pose chaining has already been applied by the caller).

    Returns:
        pixels: (N, 2) float array of (u, v) pixel coordinates.
        depths: (N,) float array of camera-frame z (depth), for all input
            points, including invalid ones.
        valid: (N,) bool array, True where depth > min_depth. Pixel/depth
            values for invalid points are returned but meaningless.
    """
    ego_to_cam = np.linalg.inv(cam_to_ego)
    points_cam = transform_points(points_ego, ego_to_cam)
    depths = points_cam[:, 2]
    valid = depths > min_depth

    safe_depths = np.clip(depths, min_depth, None)
    pixel_homogeneous = points_cam @ cam_intrinsic.T
    pixels = pixel_homogeneous[:, :2] / safe_depths[:, None]
    return pixels, depths, valid


def generate_frustum(
    image_size: tuple[int, int],
    feature_size: tuple[int, int],
    depth_bins: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    """Precompute the lift-splat frustum: camera-frame 3D coordinates for
    every (depth bin, feature row, feature col) location.

    `image_size` is the (H, W) the intrinsic is defined for; `feature_size`
    is the (fH, fW) of the backbone's output feature map. Pixel centers are
    scaled from feature-map resolution up to image resolution before
    unprojection.

    Returns:
        (D, fH, fW, 3) array of camera-frame (x, y, z) points, where z of
        every point at depth-bin index d equals depth_bins[d] exactly.
    """
    img_h, img_w = image_size
    feat_h, feat_w = feature_size
    depth_bins = np.asarray(depth_bins, dtype=np.float64)

    x_scale = img_w / feat_w
    y_scale = img_h / feat_h
    xs = (np.arange(feat_w, dtype=np.float64) + 0.5) * x_scale
    ys = (np.arange(feat_h, dtype=np.float64) + 0.5) * y_scale

    grid_d, grid_y, grid_x = np.meshgrid(depth_bins, ys, xs, indexing="ij")
    pixels_homogeneous = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1)

    inv_intrinsic = np.linalg.inv(intrinsic)
    rays = pixels_homogeneous @ inv_intrinsic.T  # z-component is 1 by construction
    points = rays * grid_d[..., None]
    return points


def camera_to_bev_indices(
    frustum_points_cam: np.ndarray,
    cam_to_ego: np.ndarray,
    grid: BEVGrid,
) -> tuple[np.ndarray, np.ndarray]:
    """Map camera-frame frustum points into BEV grid cell indices.

    Args:
        frustum_points_cam: (..., 3) camera-frame points, e.g. the output
            of `generate_frustum`.
        cam_to_ego: 4x4 camera-to-ego transform.
        grid: target BEV grid definition.

    Returns:
        indices: (..., 2) int array of (ix, iy) grid cell indices. Values
            for invalid (out-of-grid) points are clamped but meaningless.
        valid: (...,) bool array, True where the point falls inside the grid.
    """
    points_ego = transform_points(frustum_points_cam, cam_to_ego)
    xmin, xmax, xres = grid.xbound
    ymin, ymax, yres = grid.ybound

    ix = np.floor((points_ego[..., 0] - xmin) / xres).astype(np.int64)
    iy = np.floor((points_ego[..., 1] - ymin) / yres).astype(np.int64)

    valid = (
        (points_ego[..., 0] >= xmin)
        & (points_ego[..., 0] < xmax)
        & (points_ego[..., 1] >= ymin)
        & (points_ego[..., 1] < ymax)
    )

    ix_clamped = np.clip(ix, 0, grid.nx - 1)
    iy_clamped = np.clip(iy, 0, grid.ny - 1)
    indices = np.stack([ix_clamped, iy_clamped], axis=-1)
    return indices, valid
