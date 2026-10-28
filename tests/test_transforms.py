"""Regression tests for augmentation/depth-map alignment.

These exist because of a real Phase 5 bug: sparse LIDAR depth maps were
passed through at raw image resolution while the color image they
supervise was resized+cropped, so `DepthLoss` was comparing predictions
for an augmented crop against depth values from a totally different
field of view (and, incidentally, a size that wasn't even divisible by
the backbone stride). `resize_crop_depth_map` fixes this by applying the
exact same geometric transform to both.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from roadvision.data.transforms import (
    AugmentationConfig,
    CameraAugmentation,
    TransformedBEVDataset,
    resize_crop_depth_map,
)


def test_resize_crop_depth_map_identity_scale_no_crop() -> None:
    depth = np.zeros((100, 200), dtype=np.float32)
    depth[40, 60] = 12.5  # (row=40, col=60)

    out = resize_crop_depth_map(depth, final_dim=(100, 200), resize_scale=1.0, crop_left=0.0, crop_top=0.0)

    assert out.shape == (100, 200)
    assert out[40, 60] == 12.5
    assert np.count_nonzero(out) == 1


def test_resize_crop_depth_map_crop_shifts_point() -> None:
    depth = np.zeros((100, 200), dtype=np.float32)
    depth[40, 60] = 12.5

    # Cropping a 50x50 window starting at (left=50, top=30) should move the
    # point to (40 - 30, 60 - 50) = (10, 10) in the cropped frame.
    out = resize_crop_depth_map(depth, final_dim=(50, 50), resize_scale=1.0, crop_left=50.0, crop_top=30.0)

    assert out.shape == (50, 50)
    assert out[10, 10] == 12.5
    assert np.count_nonzero(out) == 1


def test_resize_crop_depth_map_resize_scales_point_location() -> None:
    depth = np.zeros((100, 200), dtype=np.float32)
    depth[50, 100] = 7.0  # dead center

    out = resize_crop_depth_map(depth, final_dim=(200, 400), resize_scale=2.0, crop_left=0.0, crop_top=0.0)

    assert out.shape == (200, 400)
    # nearest-neighbor resize maps (50, 100) -> approximately (100, 200)
    assert out[100, 200] == 7.0


class _TinyRawDataset(Dataset):
    """Minimal stand-in for `NuScenesBEVDataset.__getitem__`'s raw output."""

    def __init__(self, raw_size: tuple[int, int]):
        self.raw_size = raw_size

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict:
        h, w = self.raw_size
        image = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
        depth = np.zeros((h, w), dtype=np.float32)
        depth[h // 2, w // 2] = 20.0
        return {
            "token": "fake-token",
            "images": {"CAM_FRONT": image},
            "intrinsics": {"CAM_FRONT": np.eye(3, dtype=np.float64)},
            "cam_to_ego": {"CAM_FRONT": np.eye(4, dtype=np.float64)},
            "depth_maps": {"CAM_FRONT": depth},
        }


def test_transformed_dataset_produces_depth_maps_matching_final_dim() -> None:
    final_dim = (32, 64)
    config = AugmentationConfig(
        resize_lim=(1.0, 1.0),
        crop_jitter_px=0,
        brightness=0.0,
        contrast=0.0,
        saturation=0.0,
        hue=0.0,
    )
    augmentation = CameraAugmentation(final_dim=final_dim, config=config, train=False)
    dataset = TransformedBEVDataset(_TinyRawDataset((128, 256)), augmentation, cameras=("CAM_FRONT",))

    item = dataset[0]

    assert item["depth_maps"]["CAM_FRONT"].shape == final_dim
    assert item["images"]["CAM_FRONT"].shape[-2:] == final_dim
