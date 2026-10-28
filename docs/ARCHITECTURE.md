# Architecture

## Why Lift-Splat-Shoot, not three separate heads

The easy way to build a "multi-task camera perception" portfolio project is
three independent heads (a depth network, a segmentation network, a
detection network) sharing an image backbone. That's multi-task in name
only — the tasks don't need each other, and nothing about the architecture
would change if you deleted any one of them.

RoadVision instead uses a single [Lift-Splat-Shoot](https://arxiv.org/abs/2008.05711)-style
pipeline where **monocular depth estimation is load-bearing, not auxiliary**:
the per-camera depth-distribution head isn't just supervised for its own
sake, it's the literal mechanism that lifts 2D image features into the 3D
ego-centric space the BEV heads read from (`src/roadvision/models/bev_pool.py`).
Remove the depth head and there is no way to place camera features in BEV
space at all. That's what makes this one coherent system rather than three
models wearing a trenchcoat, and it's also why the camera intrinsics/
extrinsics (`generate_frustum_torch`, `transform_points_torch` in
`bev_pool.py`, built on the pure geometry in `src/roadvision/data/geometry.py`)
are first-class inputs to the forward pass, not metadata used only at
dataset-build time.

The alternative considered was a transformer-based BEV lifter (BEVFormer-style
deformable cross-attention from BEV queries to image features). That design
doesn't need an explicit depth estimate at all — attention implicitly learns
where to look — which is a legitimate approach but sacrifices exactly the
property this project wants to demonstrate: an explicit, auditable geometric
reasoning step with its own ground truth (LIDAR-projected depth) and its own
metrics. LSS was chosen because it's geometrically explicit end to end.

## Pipeline

```
6x (B, 3, 256, 704) camera images
        |  ImageBackbone (ResNet-18/34, truncated at layer3, stride 16)
        v
(B*6, 512, 16, 44) per-camera features          [backbone.py]
        |  DepthHead: 1 conv -> split into depth logits / context
        v
(B,6,41,16,44) depth_dist (softmax over 41 bins)
(B,6,64,16,44) context                           [depth.py]
        |  outer product: depth_dist (x) context -> per-frustum-point features
        |  generate_frustum_torch: camera-frame (u,v,depth) -> (x,y,z)
        |  transform_points_torch: camera frame -> ego frame (cam_to_ego)
        |  points_to_bev_indices_torch: ego (x,y) -> BEV cell (ix, iy)
        |  scatter-add (index_add_) into the shared grid, summed over camera+depth+height
        v
(B, 64, 200, 200) pooled BEV grid                 [bev_pool.py]
        |  BEVEncoder: ResNet BasicBlock stages, stride [1,2,2]
        v
(B, 256, 50, 50) bev_features                     [bev_encoder.py]
        |                       |
        v                       v
SegmentationHead          DetectionHead
(B, 2, 50, 50) logits     {heatmap: (B,10,50,50), offset/dims/yaw/velocity: (B,2,50,50) each}
[segmentation_head.py]    [detection_head.py]
```

`RoadVisionNet.forward` (`src/roadvision/models/roadvision_net.py`) returns
`{"bev_features", "depth_dist", "seg_logits", "det"}` — `depth_dist` is kept
in the output specifically so `DepthLoss` can supervise it directly, even
though it's also consumed internally by `bev_pool`.

**The 200x200 -> 50x50 resolution drop matters operationally**: the BEV
encoder downsamples 4x (`num_blocks: [2,2,2]`, stride 1 then stride 2 twice),
so segmentation/detection ground truth must be rasterized at that coarser
50x50 grid, not the 200x200 pooling grid. `src/roadvision/training/config.py::output_bev_grid()`
computes this derived grid from `bev_encoder.num_blocks` so it can never
silently drift out of sync with the encoder architecture.

## Tensor shapes through the pipeline (default config)

| Stage | Shape |
|---|---|
| Input images | `(B, 6, 3, 256, 704)` |
| Backbone features | `(B*6, 512, 16, 44)` |
| Depth distribution | `(B, 6, 41, 16, 44)` |
| Context features | `(B, 6, 64, 16, 44)` |
| Pooled BEV grid | `(B, 64, 200, 200)` |
| Encoded BEV features | `(B, 256, 50, 50)` |
| Segmentation logits | `(B, 2, 50, 50)` |
| Detection heatmap | `(B, 10, 50, 50)` |
| Detection offset/dims/yaw/velocity | `(B, 2, 50, 50)` each |

BEV grid is ego-frame, x-forward / y-left / z-up (ISO 8855), extent
`[-50, 50]m` on both axes at 0.5m resolution at the pooling stage (200x200),
2m effective resolution at the head stage (50x50). Depth bins span 4-45m at
1m resolution (41 bins) — see `configs/data/nuscenes_mini.yaml`.

## Design choices in the heads

- **Segmentation is multi-label sigmoid, not softmax.** `lane` sits on top
  of `drivable_area` in nuScenes' map layers, so the two classes are not
  mutually exclusive — a softmax-over-classes design would be structurally
  wrong here, not just a minor accuracy loss.
- **Detection regresses yaw as (sin, cos), not a raw angle**, avoiding the
  ±π wraparound discontinuity that makes raw-angle regression unstable near
  the boundary.
- **Detection only regresses what `bev_rasterize.py` actually supplies**:
  BEV footprint (center, length, width, yaw) + planar velocity. No
  height/z-center regression — this is a 2D BEV detector, not a full 3D box
  detector, and the docs/code are explicit about that rather than implying
  more capability than exists.
- **The detection heatmap head's final bias is initialized to `sigmoid(-2.19) ≈ 0.1`**
  (CenterNet convention), since the overwhelming majority of BEV cells are
  background — starting from an unbiased 0.5 would make early focal-loss
  training unnecessarily unstable.

## Why uncertainty-weighted multi-task loss

`src/roadvision/losses/multitask_loss.py` implements Kendall, Gal & Cipolla
(2018)'s homoscedastic uncertainty weighting: each task owns a learned
`log_var_i`, and the total loss is `sum_i exp(-log_var_i) * L_i + log_var_i`.
This was chosen over fixed hand-tuned weights for a concrete reason: the
three losses here are not on comparable scales or difficulty curves —
depth cross-entropy over 41 bins, detection's Gaussian-focal heatmap loss,
and segmentation's class-balanced focal loss all live in different numeric
ranges and converge at different rates. Hand-picked fixed weights would need
re-tuning any time the backbone, loss formulation, or dataset mix changes,
and silently produce a model that's implicitly over- or under-fit to
whichever task's raw loss happens to be numerically largest. The learned
weighting re-balances automatically as each task's relative difficulty
shifts over training, and the resulting `uncertainty/<task>` values
(logged by `trackers.py`) are themselves a useful training diagnostic — if
one task's learned uncertainty collapses toward zero while another's
balloons, that's a legible signal, not a silent imbalance.
