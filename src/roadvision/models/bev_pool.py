"""Lift-splat: lift per-camera depth-distributed features into 3D, then
splat them into a shared ego-centric BEV grid.

Frustum generation and the camera->ego->BEV-index geometry here are a
batched-torch mirror of the pure numpy functions in
`roadvision.data.geometry` (`generate_frustum`, `camera_to_bev_indices`).
They can't literally call the numpy versions since this runs inside
autograd on batched GPU tensors with per-sample (post-augmentation)
intrinsics, but `tests/test_bev_pool.py` cross-checks both implementations
against each other on identical inputs to keep them from drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from roadvision.data.geometry import BEVGrid


@dataclass(frozen=True)
class FrustumConfig:
    image_size: tuple[int, int]  # (H, W) the intrinsic is defined for
    feature_size: tuple[int, int]  # (fH, fW) of the backbone feature map


def generate_frustum_torch(
    frustum_config: FrustumConfig,
    depth_bins: torch.Tensor,
    intrinsic: torch.Tensor,
) -> torch.Tensor:
    """Batched counterpart of `geometry.generate_frustum`.

    Args:
        depth_bins: (D,)
        intrinsic: (..., 3, 3), one per camera/sample (post-augmentation).

    Returns:
        (..., D, fH, fW, 3) camera-frame points.
    """
    img_h, img_w = frustum_config.image_size
    feat_h, feat_w = frustum_config.feature_size
    device, dtype = intrinsic.device, intrinsic.dtype

    x_scale, y_scale = img_w / feat_w, img_h / feat_h
    xs = (torch.arange(feat_w, device=device, dtype=dtype) + 0.5) * x_scale
    ys = (torch.arange(feat_h, device=device, dtype=dtype) + 0.5) * y_scale
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (fH, fW)
    ones = torch.ones_like(grid_x)
    pixels = torch.stack([grid_x, grid_y, ones], dim=-1)  # (fH, fW, 3)

    leading_shape = intrinsic.shape[:-2]
    pixels = pixels.expand(*leading_shape, feat_h, feat_w, 3)
    inv_intrinsic = torch.inverse(intrinsic)
    rays = torch.einsum("...ij,...hwj->...hwi", inv_intrinsic, pixels)  # (..., fH, fW, 3), z=1

    rays = rays.unsqueeze(-4)  # (..., 1, fH, fW, 3)
    bins = depth_bins.view(*([1] * len(leading_shape)), -1, 1, 1, 1)
    return rays * bins  # (..., D, fH, fW, 3)


def transform_points_torch(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Apply a batched 4x4 homogeneous transform to (..., 3) points.

    `transform`'s batch dims must broadcast against a prefix of `points`'s
    batch dims (e.g. transform (B, N, 4, 4) against points (B, N, D, H, W, 3)).
    """
    extra_dims = points.dim() - transform.dim() + 1
    rot = transform[..., :3, :3]
    trans = transform[..., :3, 3]
    for _ in range(extra_dims):
        rot = rot.unsqueeze(-3)
        trans = trans.unsqueeze(-2)
    return torch.einsum("...ij,...j->...i", rot, points) + trans


def points_to_bev_indices_torch(
    points_ego: torch.Tensor, grid: BEVGrid
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched counterpart of `geometry.camera_to_bev_indices`.

    Returns:
        indices: (..., 2) long tensor of clamped (ix, iy).
        valid: (...,) bool tensor, True where the point falls inside the grid.
    """
    xmin, xmax, xres = grid.xbound
    ymin, ymax, yres = grid.ybound

    ix = torch.floor((points_ego[..., 0] - xmin) / xres).long()
    iy = torch.floor((points_ego[..., 1] - ymin) / yres).long()

    valid = (
        (points_ego[..., 0] >= xmin)
        & (points_ego[..., 0] < xmax)
        & (points_ego[..., 1] >= ymin)
        & (points_ego[..., 1] < ymax)
    )
    ix = ix.clamp(0, grid.nx - 1)
    iy = iy.clamp(0, grid.ny - 1)
    return torch.stack([ix, iy], dim=-1), valid


class BEVPool(nn.Module):
    """Lifts per-camera (depth_dist, context) into frustum-point features
    and splats them (scatter-add, summed over depth/height) into a shared
    ego-centric BEV grid.

    Correctness-first implementation: one `index_add_` per batch sample.
    The classic LSS "cumsum trick" (sort points by BEV cell, cumsum, then
    difference) avoids materializing the full (N_cam*D*fH*fW, C) point
    tensor and is the standard optimization here, but is not needed for
    this model to be correct, only faster -- left as a documented follow-up.
    """

    def __init__(self, frustum_config: FrustumConfig, depth_bins: torch.Tensor, grid: BEVGrid):
        super().__init__()
        self.frustum_config = frustum_config
        self.register_buffer("depth_bins", depth_bins, persistent=False)
        self.grid = grid

    def forward(
        self,
        depth_dist: torch.Tensor,
        context: torch.Tensor,
        intrinsic: torch.Tensor,
        cam_to_ego: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            depth_dist: (B, N, D, fH, fW)
            context: (B, N, C, fH, fW)
            intrinsic: (B, N, 3, 3)
            cam_to_ego: (B, N, 4, 4)

        Returns:
            (B, C, grid.ny, grid.nx) BEV feature grid.
        """
        batch_size, num_channels = context.shape[0], context.shape[2]

        frustum_cam = generate_frustum_torch(self.frustum_config, self.depth_bins, intrinsic)  # (B,N,D,fH,fW,3)
        frustum_ego = transform_points_torch(frustum_cam, cam_to_ego)
        indices, valid = points_to_bev_indices_torch(frustum_ego, self.grid)  # (B,N,D,fH,fW,2/-)

        # (B,N,D,fH,fW,1) * (B,N,1,fH,fW,C) -> (B,N,D,fH,fW,C)
        features = depth_dist.unsqueeze(-1) * context.permute(0, 1, 3, 4, 2).unsqueeze(2)

        bev = depth_dist.new_zeros(batch_size, self.grid.ny * self.grid.nx, num_channels)
        flat_index = indices[..., 1] * self.grid.nx + indices[..., 0]  # row-major (iy, ix)

        for b in range(batch_size):
            mask_b = valid[b].reshape(-1)
            idx_b = flat_index[b].reshape(-1)[mask_b]
            feat_b = features[b].reshape(-1, num_channels)[mask_b]
            bev[b].index_add_(0, idx_b, feat_b)

        return bev.view(batch_size, self.grid.ny, self.grid.nx, num_channels).permute(0, 3, 1, 2).contiguous()
