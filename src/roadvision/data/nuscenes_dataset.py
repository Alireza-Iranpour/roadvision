"""PyTorch Dataset wrapping nuscenes-devkit for multi-camera BEV training.

`nuscenes-devkit` (and the actual downloaded dataset) is only required at
instantiation time, not at import time, so this module and the pure
geometry it depends on stay unit-testable without either installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset

from roadvision.data.bev_rasterize import (
    DETECTION_CLASSES,
    SEGMENTATION_LAYERS,
    NuScenesBEVAnnotations,
    rasterize_boxes_to_bev,
    rasterize_polygons_to_bev,
)
from roadvision.data.geometry import BEVGrid, project_lidar_to_camera, transform_points

CAMERA_NAMES = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
    "CAM_BACK_LEFT",
)


@dataclass
class Sample:
    """One multi-camera training sample, pre-geometry-processing."""

    token: str
    image_paths: dict[str, str]
    intrinsics: dict[str, np.ndarray]  # cam_name -> (3, 3)
    cam_to_ego: dict[str, np.ndarray]  # cam_name -> (4, 4)
    lidar_points_ego: np.ndarray  # (N, 3), in ego frame at this sample's timestamp


def pose_record_to_matrix(rotation_wxyz: list[float], translation: list[float]) -> np.ndarray:
    """Build a 4x4 transform from a nuScenes {rotation, translation} record.

    nuScenes stores rotation as a wxyz quaternion and a translation vector
    describing "this sensor/ego frame expressed in its parent frame" --
    i.e. exactly the convention this module calls `*_to_ego` / `ego_to_global`.
    """
    transform = np.eye(4)
    transform[:3, :3] = Quaternion(rotation_wxyz).rotation_matrix
    transform[:3, 3] = translation
    return transform


def _load_nuscenes_devkit():
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError as exc:  # pragma: no cover - exercised only without the dep installed
        raise ImportError(
            "nuscenes-devkit is required to instantiate NuScenesBEVDataset. "
            "Install it with `pip install nuscenes-devkit` and download v1.0-mini "
            "via scripts/download_nuscenes_mini.sh."
        ) from exc
    return NuScenes


class NuScenesBEVDataset(Dataset):
    """Multi-camera nuScenes samples with LIDAR-derived per-camera depth.

    Each item returns raw (un-augmented, un-collated) numpy/PIL data; image
    augmentation and tensor collation happen in `transforms.py` so this
    class stays a thin, testable data-access layer.
    """

    def __init__(
        self,
        dataroot: str,
        version: str = "v1.0-mini",
        cameras: tuple[str, ...] = CAMERA_NAMES,
        verbose: bool = False,
        output_grid: BEVGrid | None = None,
        scene_names: tuple[str, ...] | None = None,
    ) -> None:
        """
        Args:
            output_grid: if given, `__getitem__` also returns `det_target`
                and `seg_target` rasterized at this grid's resolution --
                which must match the *model output* resolution (after the
                BEV encoder's downsampling), not `data.bev_grid`. If None,
                only images/depth are returned (sufficient for Phase-2-only
                depth/BEV-pooling work).
            scene_names: if given, restrict samples to only these scenes
                (by `scene["name"]`, e.g. "scene-0061") -- this is how
                `training/train.py` and `eval/evaluate.py` build disjoint
                train/val splits, since nuScenes-mini has no official one.
        """
        NuScenes = _load_nuscenes_devkit()
        self.dataroot = Path(dataroot)
        self.cameras = cameras
        self.nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=verbose)

        if scene_names is not None:
            allowed_scene_tokens = {s["token"] for s in self.nusc.scene if s["name"] in scene_names}
            missing = set(scene_names) - {s["name"] for s in self.nusc.scene}
            if missing:
                raise ValueError(f"scene_names not found in this dataset version: {sorted(missing)}")
            self.sample_tokens = [
                s["token"] for s in self.nusc.sample if s["scene_token"] in allowed_scene_tokens
            ]
        else:
            self.sample_tokens = [s["token"] for s in self.nusc.sample]

        self.output_grid = output_grid
        self.annotations = self._build_annotations() if output_grid is not None else None

    def _build_annotations(self) -> NuScenesBEVAnnotations:
        from nuscenes.map_expansion.map_api import NuScenesMap

        locations = {self.nusc.get("log", scene["log_token"])["location"] for scene in self.nusc.scene}
        nusc_map_by_location = {
            location: NuScenesMap(dataroot=str(self.dataroot), map_name=location) for location in locations
        }
        return NuScenesBEVAnnotations(self.nusc, nusc_map_by_location)

    def _sample_location(self, sample: dict[str, Any]) -> str:
        scene = self.nusc.get("scene", sample["scene_token"])
        return self.nusc.get("log", scene["log_token"])["location"]

    def __len__(self) -> int:
        return len(self.sample_tokens)

    def __getitem__(self, index: int) -> dict[str, Any]:
        token = self.sample_tokens[index]
        sample = self.nusc.get("sample", token)

        images: dict[str, Image.Image] = {}
        intrinsics: dict[str, np.ndarray] = {}
        cam_to_ego: dict[str, np.ndarray] = {}

        for cam in self.cameras:
            cam_token = sample["data"][cam]
            cam_data = self.nusc.get("sample_data", cam_token)
            calib = self.nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])

            images[cam] = Image.open(self.dataroot / cam_data["filename"]).convert("RGB")
            intrinsics[cam] = np.array(calib["camera_intrinsic"], dtype=np.float64)
            cam_to_ego[cam] = pose_record_to_matrix(calib["rotation"], calib["translation"])

        lidar_points_ego = self._lidar_points_in_ego_frame(sample, reference_cam=self.cameras[0])

        depth_maps = {
            cam: self._project_depth_map(lidar_points_ego, intrinsics[cam], cam_to_ego[cam], images[cam].size)
            for cam in self.cameras
        }

        item = {
            "token": token,
            "images": images,
            "intrinsics": intrinsics,
            "cam_to_ego": cam_to_ego,
            "depth_maps": depth_maps,
        }
        if self.annotations is not None:
            item["det_target"], item["seg_target"] = self._build_targets(sample, token)
        return item

    def _build_targets(self, sample: dict[str, Any], token: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
        grid = self.output_grid
        assert grid is not None and self.annotations is not None

        boxes = self.annotations.get_detection_boxes(token)
        det = rasterize_boxes_to_bev(boxes, grid, num_classes=len(DETECTION_CLASSES))
        det_target = {
            "heatmap": det.heatmap,
            "offset": det.offset,
            "dims": det.dims,
            "yaw": det.yaw,
            "velocity": det.velocity,
            "mask": det.mask,
        }

        location = self._sample_location(sample)
        layers = self.annotations.get_segmentation_polygons(token, grid, location)
        seg_target = np.stack(
            [rasterize_polygons_to_bev(layers[layer_name], grid) for layer_name in SEGMENTATION_LAYERS], axis=0
        )
        return det_target, seg_target

    def _lidar_points_in_ego_frame(self, sample: dict[str, Any], reference_cam: str) -> np.ndarray:
        """Return LIDAR points expressed in the ego frame at `reference_cam`'s
        timestamp, chaining through the global frame to account for the
        lidar and camera sampling at (slightly) different times.
        """
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = self.nusc.get("sample_data", lidar_token)
        lidar_calib = self.nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
        lidar_ego_pose = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])

        from nuscenes.utils.data_classes import LidarPointCloud

        cloud = LidarPointCloud.from_file(str(self.dataroot / lidar_data["filename"]))
        points_lidar = cloud.points[:3, :].T  # (N, 3)

        lidar_to_ego = pose_record_to_matrix(lidar_calib["rotation"], lidar_calib["translation"])
        lidar_ego_to_global = pose_record_to_matrix(lidar_ego_pose["rotation"], lidar_ego_pose["translation"])

        cam_token = sample["data"][reference_cam]
        cam_data = self.nusc.get("sample_data", cam_token)
        cam_ego_pose = self.nusc.get("ego_pose", cam_data["ego_pose_token"])
        cam_ego_to_global = pose_record_to_matrix(cam_ego_pose["rotation"], cam_ego_pose["translation"])
        global_to_cam_ego = np.linalg.inv(cam_ego_to_global)

        lidar_to_target_ego = global_to_cam_ego @ lidar_ego_to_global @ lidar_to_ego
        return transform_points(points_lidar, lidar_to_target_ego)

    @staticmethod
    def _project_depth_map(
        points_ego: np.ndarray,
        intrinsic: np.ndarray,
        cam_to_ego: np.ndarray,
        image_size: tuple[int, int],
    ) -> np.ndarray:
        """Sparse (H, W) depth map: 0 where no LIDAR return projects, else depth."""
        width, height = image_size
        pixels, depths, valid = project_lidar_to_camera(points_ego, intrinsic, cam_to_ego)

        u = np.round(pixels[:, 0]).astype(np.int64)
        v = np.round(pixels[:, 1]).astype(np.int64)
        in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        keep = valid & in_bounds

        depth_map = np.zeros((height, width), dtype=np.float32)
        depth_map[v[keep], u[keep]] = depths[keep]
        return depth_map
