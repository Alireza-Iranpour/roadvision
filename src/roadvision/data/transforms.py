"""Image augmentation and batch collation for multi-camera BEV samples.

NOTE on horizontal flip: BEV detection/segmentation targets encode an
absolute left/right (ego +y) position and a signed yaw. Flipping only the
camera images mirrors the scene in image space but leaves those targets
(and the camera extrinsics) unchanged, producing an inconsistent sample.
A correct implementation must also mirror cam_to_ego (negate the y-axis
and re-derive yaw) and every BEV target in lockstep. That is not
implemented here; `horizontal_flip_prob` must stay 0.0 until it is.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


@dataclass(frozen=True)
class AugmentationConfig:
    resize_lim: tuple[float, float]
    crop_jitter_px: int
    brightness: float
    contrast: float
    saturation: float
    hue: float
    horizontal_flip_prob: float = 0.0

    def __post_init__(self) -> None:
        if self.horizontal_flip_prob != 0.0:
            raise ValueError(
                "horizontal_flip_prob must be 0.0: flipping requires mirroring "
                "BEV targets and camera extrinsics, which is not implemented."
            )


class CameraAugmentation:
    """Per-image resize + random crop + photometric jitter, applied
    independently per camera (each camera keeps its own random resize/crop
    so the model doesn't learn a fixed camera-index-to-crop shortcut)."""

    def __init__(self, final_dim: tuple[int, int], config: AugmentationConfig, train: bool = True):
        self.final_h, self.final_w = final_dim
        self.config = config
        self.train = train
        self.color_jitter = T.ColorJitter(
            brightness=config.brightness,
            contrast=config.contrast,
            saturation=config.saturation,
            hue=config.hue,
        )
        self.to_tensor = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __call__(self, image: Image.Image) -> tuple[torch.Tensor, dict[str, float]]:
        src_w, src_h = image.size

        if self.train:
            resize_scale = random.uniform(*self.config.resize_lim)
        else:
            resize_scale = sum(self.config.resize_lim) / 2.0

        resized_w = int(src_w * resize_scale)
        resized_h = int(src_h * resize_scale)
        resized_w = max(resized_w, self.final_w)
        resized_h = max(resized_h, self.final_h)
        image = image.resize((resized_w, resized_h), Image.BILINEAR)

        max_left = resized_w - self.final_w
        max_top = resized_h - self.final_h
        if self.train:
            jitter = self.config.crop_jitter_px
            left = min(max(0, max_left // 2 + random.randint(-jitter, jitter)), max_left)
            top = min(max(0, max_top // 2 + random.randint(-jitter, jitter)), max_top)
        else:
            left, top = max_left // 2, max_top // 2

        image = image.crop((left, top, left + self.final_w, top + self.final_h))
        if self.train:
            image = self.color_jitter(image)

        tensor = self.to_tensor(image)
        crop_params = {
            "resize_scale": resize_scale,
            "crop_left": float(left),
            "crop_top": float(top),
        }
        return tensor, crop_params


def resize_crop_depth_map(
    depth_map: np.ndarray, final_dim: tuple[int, int], resize_scale: float, crop_left: float, crop_top: float
) -> np.ndarray:
    """Apply the same resize+crop `CameraAugmentation` applied to the color
    image to a sparse LIDAR depth map, so depth pixel (u, v) in the returned
    (H, W) == `final_dim` array lines up with the same real-world point as
    color pixel (u, v) -- without this, the depth map stays in raw-image
    pixel space while the image it's meant to supervise has been reframed,
    and the two silently point at different parts of the scene.

    Resized with nearest-neighbor, not bilinear: averaging a mostly-zero
    sparse map would blend real depths with "no return" zeros into bogus
    intermediate values.
    """
    final_h, final_w = final_dim
    src_h, src_w = depth_map.shape
    resized_w, resized_h = max(int(src_w * resize_scale), final_w), max(int(src_h * resize_scale), final_h)

    image = Image.fromarray(depth_map, mode="F").resize((resized_w, resized_h), Image.NEAREST)
    left, top = int(crop_left), int(crop_top)
    image = image.crop((left, top, left + final_w, top + final_h))
    return np.array(image, dtype=np.float32)


def adjust_intrinsic_for_crop(intrinsic: np.ndarray, resize_scale: float, crop_left: float, crop_top: float) -> np.ndarray:
    """Update a camera intrinsic matrix to match a resize+crop applied to
    its image, so frustum geometry stays consistent with the augmented
    pixels actually fed to the backbone."""
    adjusted = intrinsic.copy()
    adjusted[0, 0] *= resize_scale  # fx
    adjusted[1, 1] *= resize_scale  # fy
    adjusted[0, 2] = intrinsic[0, 2] * resize_scale - crop_left  # cx
    adjusted[1, 2] = intrinsic[1, 2] * resize_scale - crop_top  # cy
    return adjusted


class TransformedBEVDataset(Dataset):
    """Wraps a raw `NuScenesBEVDataset` item (PIL images + full-res depth
    maps + optional det/seg targets) with per-camera augmentation, producing
    exactly the per-sample tensor dict `collate_bev_batch` expects.

    Depth maps are resized+cropped to `final_dim` with the *same*
    resize_scale/crop as the color image they correspond to (nearest-
    neighbor, via `resize_crop_depth_map`), so depth pixel (u, v) and color
    pixel (u, v) describe the same point after augmentation. `DepthLoss`
    then only needs to downsample from `final_dim` to feature-map
    resolution by a fixed stride, using the *sparse-aware* nearest-valid-
    depth pooling in `losses/depth_loss.py`.
    """

    def __init__(self, base_dataset: Dataset, augmentation: CameraAugmentation, cameras: tuple[str, ...]):
        self.base_dataset = base_dataset
        self.augmentation = augmentation
        self.cameras = cameras
        self.final_dim = (augmentation.final_h, augmentation.final_w)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw = self.base_dataset[index]

        images: dict[str, torch.Tensor] = {}
        intrinsics: dict[str, np.ndarray] = {}
        depth_maps: dict[str, np.ndarray] = {}
        for cam in self.cameras:
            tensor, crop_params = self.augmentation(raw["images"][cam])
            images[cam] = tensor
            intrinsics[cam] = adjust_intrinsic_for_crop(raw["intrinsics"][cam], **_crop_kwargs(crop_params))
            depth_maps[cam] = resize_crop_depth_map(raw["depth_maps"][cam], self.final_dim, **_crop_kwargs(crop_params))

        item = {
            "token": raw["token"],
            "images": images,
            "intrinsics": intrinsics,
            "cam_to_ego": raw["cam_to_ego"],
            "depth_maps": depth_maps,
        }
        if "det_target" in raw:
            item["det_target"] = raw["det_target"]
        if "seg_target" in raw:
            item["seg_target"] = raw["seg_target"]
        return item


def _crop_kwargs(crop_params: dict[str, float]) -> dict[str, float]:
    return {
        "resize_scale": crop_params["resize_scale"],
        "crop_left": crop_params["crop_left"],
        "crop_top": crop_params["crop_top"],
    }


def collate_bev_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack a list of per-sample dicts (each already augmented to fixed-size
    per-camera tensors) into batched tensors, keyed the same way but with a
    leading batch dimension and camera order fixed by `cameras`."""
    cameras = list(samples[0]["images"].keys())

    images = torch.stack(
        [torch.stack([sample["images"][cam] for cam in cameras], dim=0) for sample in samples], dim=0
    )  # (B, num_cams, 3, H, W)

    intrinsics = torch.stack(
        [
            torch.stack([torch.from_numpy(sample["intrinsics"][cam]).float() for cam in cameras], dim=0)
            for sample in samples
        ],
        dim=0,
    )  # (B, num_cams, 3, 3)

    cam_to_ego = torch.stack(
        [
            torch.stack([torch.from_numpy(sample["cam_to_ego"][cam]).float() for cam in cameras], dim=0)
            for sample in samples
        ],
        dim=0,
    )  # (B, num_cams, 4, 4)

    depth_maps = torch.stack(
        [
            torch.stack([torch.from_numpy(sample["depth_maps"][cam]).float() for cam in cameras], dim=0)
            for sample in samples
        ],
        dim=0,
    )  # (B, num_cams, H, W)

    batch = {
        "tokens": [sample["token"] for sample in samples],
        "images": images,
        "intrinsics": intrinsics,
        "cam_to_ego": cam_to_ego,
        "depth_maps": depth_maps,
    }

    if "det_target" in samples[0]:
        batch["det_target"] = {
            key: torch.stack([torch.from_numpy(sample["det_target"][key]).float() for sample in samples], dim=0)
            for key in samples[0]["det_target"]
        }
    if "seg_target" in samples[0]:
        batch["seg_target"] = torch.stack(
            [torch.from_numpy(sample["seg_target"]).float() for sample in samples], dim=0
        )
    return batch
