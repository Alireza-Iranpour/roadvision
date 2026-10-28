"""BEV ground-truth rasterization for the detection and segmentation heads.

The rasterization functions (`rasterize_boxes_to_bev`,
`rasterize_polygons_to_bev`) take plain ego-frame boxes/polygons and are
unit-testable without nuscenes-devkit. `NuScenesBEVAnnotations` is the thin
devkit-dependent adapter that turns a sample token into those plain
structures (3D box annotations -> `BEVBox`, map-expansion-API layers ->
polygon coordinate lists) -- it is exercised against real data/maps, not
unit tests, same as `NuScenesBEVDataset`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from pyquaternion import Quaternion

from roadvision.data.geometry import BEVGrid

# 10 nuScenes detection classes, in a fixed order matching
# configs/model/roadvision_net.yaml:detection_head.classes.
DETECTION_CLASSES = (
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic_cone",
    "barrier",
)

# nuScenes raw category names are finer-grained (e.g.
# "vehicle.car", "human.pedestrian.adult") and map many-to-one onto the
# above. Prefixes are matched longest-first.
_CATEGORY_NAME_TO_CLASS: dict[str, str] = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus": "bus",
    "vehicle.trailer": "trailer",
    "vehicle.construction": "construction_vehicle",
    "human.pedestrian": "pedestrian",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.barrier": "barrier",
}

SEGMENTATION_LAYERS = ("drivable_area", "lane")


@dataclass(frozen=True)
class BEVBox:
    """A single object box, in ego-frame BEV coordinates (meters, radians)."""

    center_x: float
    center_y: float
    length: float
    width: float
    yaw: float  # radians, 0 = facing ego +x, increasing counter-clockwise
    class_idx: int
    velocity: tuple[float, float] = (0.0, 0.0)


@dataclass
class DetectionTargets:
    heatmap: np.ndarray  # (num_classes, ny, nx), in [0, 1]
    offset: np.ndarray  # (2, ny, nx), sub-cell center offset in [0, 1)
    dims: np.ndarray  # (2, ny, nx), (log length_m, log width_m)
    yaw: np.ndarray  # (2, ny, nx), (sin, cos)
    velocity: np.ndarray  # (2, ny, nx), (vx, vy) m/s in ego frame
    mask: np.ndarray  # (ny, nx), 1 at object center cells, else 0


def category_name_to_class_idx(category_name: str) -> int | None:
    """Map a raw nuScenes category name to a DETECTION_CLASSES index, or
    None if it's not one of the classes we train on (e.g. "animal")."""
    for prefix, class_name in sorted(_CATEGORY_NAME_TO_CLASS.items(), key=lambda kv: -len(kv[0])):
        if category_name.startswith(prefix):
            return DETECTION_CLASSES.index(class_name)
    return None


def _gaussian_radius(length_cells: float, width_cells: float, min_radius: float = 1.0) -> float:
    """CenterNet-style heuristic: Gaussian radius scaled to the box's
    footprint in grid cells, floored so tiny/degenerate boxes still render
    a usable peak."""
    return max(min_radius, min(length_cells, width_cells) / 2.0)


def _splat_gaussian(heatmap: np.ndarray, cx: int, cy: int, radius: float) -> None:
    ny, nx = heatmap.shape
    sigma = max(radius / 3.0, 1e-3)
    extent = max(int(radius), 1)
    for dy in range(-extent, extent + 1):
        y = cy + dy
        if y < 0 or y >= ny:
            continue
        for dx in range(-extent, extent + 1):
            x = cx + dx
            if x < 0 or x >= nx:
                continue
            value = np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
            heatmap[y, x] = max(heatmap[y, x], value)


def rasterize_boxes_to_bev(boxes: Iterable[BEVBox], grid: BEVGrid, num_classes: int) -> DetectionTargets:
    """Rasterize object boxes into CenterNet-style BEV detection targets."""
    ny, nx = grid.ny, grid.nx
    xmin, _, xres = grid.xbound
    ymin, _, yres = grid.ybound

    heatmap = np.zeros((num_classes, ny, nx), dtype=np.float32)
    offset = np.zeros((2, ny, nx), dtype=np.float32)
    dims = np.zeros((2, ny, nx), dtype=np.float32)
    yaw = np.zeros((2, ny, nx), dtype=np.float32)
    velocity = np.zeros((2, ny, nx), dtype=np.float32)
    mask = np.zeros((ny, nx), dtype=np.float32)

    for box in boxes:
        fx = (box.center_x - xmin) / xres
        fy = (box.center_y - ymin) / yres
        cx, cy = int(np.floor(fx)), int(np.floor(fy))
        if not (0 <= cx < nx and 0 <= cy < ny):
            continue

        radius = _gaussian_radius(box.length / xres, box.width / yres)
        _splat_gaussian(heatmap[box.class_idx], cx, cy, radius)

        offset[:, cy, cx] = [fx - cx, fy - cy]
        dims[:, cy, cx] = [np.log(box.length), np.log(box.width)]
        yaw[:, cy, cx] = [np.sin(box.yaw), np.cos(box.yaw)]
        velocity[:, cy, cx] = box.velocity
        mask[cy, cx] = 1.0

    return DetectionTargets(heatmap=heatmap, offset=offset, dims=dims, yaw=yaw, velocity=velocity, mask=mask)


def rasterize_polygons_to_bev(polygons: Iterable[list[tuple[float, float]]], grid: BEVGrid) -> np.ndarray:
    """Binary BEV occupancy mask for ego-frame polygons (e.g. a nuScenes map
    layer's drivable-area/lane polygons, already expressed in ego-relative
    meters by the caller)."""
    from matplotlib.path import Path

    ny, nx = grid.ny, grid.nx
    xmin, _, xres = grid.xbound
    ymin, _, yres = grid.ybound

    polys = [poly for poly in polygons if len(poly) >= 3]
    if not polys:
        return np.zeros((ny, nx), dtype=np.float32)

    xs = xmin + (np.arange(nx) + 0.5) * xres
    ys = ymin + (np.arange(ny) + 0.5) * yres
    grid_x, grid_y = np.meshgrid(xs, ys)
    query_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)

    mask = np.zeros(query_points.shape[0], dtype=bool)
    for poly in polys:
        mask |= Path(poly).contains_points(query_points)
    return mask.reshape(ny, nx).astype(np.float32)


class NuScenesBEVAnnotations:
    """Devkit-dependent adapter: sample token -> BEVBox list / segmentation
    polygons, in the ego frame at the sample's LIDAR_TOP timestamp (the
    same reference frame `NuScenesBEVDataset` uses for depth projection).
    """

    def __init__(self, nusc: Any, nusc_map_by_location: dict[str, Any]):
        self.nusc = nusc
        self.nusc_map_by_location = nusc_map_by_location

    def get_detection_boxes(self, sample_token: str) -> list[BEVBox]:
        """Box centers/yaw/velocity in the ego frame, following the same
        translate-then-rotate-by-inverse-ego-pose idiom nuscenes-devkit's
        own eval code uses to go from global to ego frame."""
        sample = self.nusc.get("sample", sample_token)
        lidar_data = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])
        ego_translation = np.array(ego_pose["translation"])
        ego_rotation_inv = Quaternion(ego_pose["rotation"]).inverse

        boxes: list[BEVBox] = []
        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            class_idx = category_name_to_class_idx(ann["category_name"])
            if class_idx is None:
                continue

            box = self.nusc.get_box(ann_token)
            box.translate(-ego_translation)
            box.rotate(ego_rotation_inv)
            yaw = _yaw_from_rotation_matrix(box.rotation_matrix)

            vx, vy, _ = self.nusc.box_velocity(ann_token)
            if np.isnan(vx) or np.isnan(vy):
                vx, vy = 0.0, 0.0
            vx, vy, _ = ego_rotation_inv.rotate(np.array([vx, vy, 0.0]))

            boxes.append(
                BEVBox(
                    center_x=float(box.center[0]),
                    center_y=float(box.center[1]),
                    length=float(box.wlh[1]),
                    width=float(box.wlh[0]),
                    yaw=yaw,
                    class_idx=class_idx,
                    velocity=(float(vx), float(vy)),
                )
            )
        return boxes

    def get_segmentation_polygons(
        self, sample_token: str, grid: BEVGrid, location: str
    ) -> dict[str, list[list[tuple[float, float]]]]:
        from roadvision.data.nuscenes_dataset import pose_record_to_matrix

        sample = self.nusc.get("sample", sample_token)
        lidar_data = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])
        ego_to_global = pose_record_to_matrix(ego_pose["rotation"], ego_pose["translation"])
        global_to_ego = np.linalg.inv(ego_to_global)

        ego_x, ego_y = ego_pose["translation"][0], ego_pose["translation"][1]
        half_extent = max(grid.xbound[1] - grid.xbound[0], grid.ybound[1] - grid.ybound[0])
        patch = (ego_x - half_extent, ego_y - half_extent, ego_x + half_extent, ego_y + half_extent)

        nusc_map = self.nusc_map_by_location[location]
        layers: dict[str, list[list[tuple[float, float]]]] = {}
        for layer_name in SEGMENTATION_LAYERS:
            records = nusc_map.get_records_in_patch(patch, [layer_name], mode="intersect")[layer_name]
            polygons_ego = []
            for record_token in records:
                record = nusc_map.get(layer_name, record_token)
                # "lane" records have a single polygon_token; "drivable_area"
                # records have a polygon_tokens list (one area can be made
                # of several disjoint/holed polygons).
                polygon_tokens = record.get("polygon_tokens") or [record["polygon_token"]]
                for polygon_token in polygon_tokens:
                    polygon = nusc_map.extract_polygon(polygon_token)
                    global_xy = np.array(polygon.exterior.coords)
                    global_xyz = np.concatenate([global_xy, np.zeros((len(global_xy), 1))], axis=1)
                    ego_xyz = (
                        global_to_ego @ np.concatenate([global_xyz, np.ones((len(global_xyz), 1))], axis=1).T
                    ).T
                    polygons_ego.append([(p[0], p[1]) for p in ego_xyz[:, :3]])
            layers[layer_name] = polygons_ego
        return layers


def _yaw_from_rotation_matrix(rotation_matrix: np.ndarray) -> float:
    """Planar yaw (rotation about ego +z) from a 3x3 rotation matrix,
    assuming the object/ego pair only differ by a z-axis rotation (true
    for nuScenes vehicle poses on near-flat ground)."""
    return float(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
