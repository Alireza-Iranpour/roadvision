# RoadVision

Multi-task bird's-eye-view (BEV) perception from multi-camera input, built
on nuScenes. A single [Lift-Splat-Shoot](https://arxiv.org/abs/2008.05711)-style
network takes six surround cameras and produces:

- **Monocular depth** per camera (supervised by LIDAR-projected depth),
  which also doubles as the mechanism that lifts 2D image features into
  the shared BEV representation — not a bolted-on auxiliary task, it's
  architecturally load-bearing
- **BEV semantic segmentation** (drivable area, lane boundaries)
- **BEV object detection** (3D boxes: position, size, yaw, velocity)

trained jointly with a learned (uncertainty-weighted) multi-task loss,
rather than as three separately-tuned heads.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full pipeline,
tensor shapes, and design rationale, and [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md)
for how the data pipeline, distributed training, serving, and monitoring
would change at production scale.

## Status

All 6 phases are complete; **53/53 tests pass**. A real training run has
been executed against downloaded nuScenes-mini (10 scenes, 404 samples) on
CPU — see [docs/RESULTS.md](docs/RESULTS.md) for the actual metrics, a real
diagnosed day/night generalization gap and class-imbalance failure mode,
the two fixes applied, and an honest account of a memory-pressure incident
that limited the before/after comparison's rigor.

| Phase | What | Status |
|---|---|---|
| 1 | Data pipeline: nuScenes dataset wrapper, camera/BEV projection geometry, BEV ground-truth rasterization | Done, unit tested |
| 2 | Model core: backbone, depth-distribution head, lift-splat BEV pooling, BEV encoder | Done, unit tested |
| 3 | Segmentation/detection heads, per-task losses, uncertainty-weighted multi-task loss, DDP/AMP/checkpointing training loop | Done, unit + smoke tested |
| 4 | Per-task metrics, failure-mode slicing, checkpoint-driven evaluation entrypoint, BEV visualizer | Done, verified against synthetic data |
| 5 | Real training run on nuScenes-mini, diagnosed failure modes, 2 targeted fixes | Done — see [docs/RESULTS.md](docs/RESULTS.md) for real metrics and caveats |
| 6 | Architecture + system design docs, README | Done |

**Honesty note:** Phase 5's real run surfaced and fixed two real bugs
(sparse depth maps misaligned with augmented images, and a per-class
detection AP evaluation bug), but a before/after comparison of the two
reliability fixes couldn't be completed at a training budget large enough
to be conclusive — see [docs/RESULTS.md](docs/RESULTS.md) for exactly what
was and wasn't demonstrated, rather than overstating it here.

## Setup

Requires Python 3.12 (torch/shapely don't yet have prebuilt wheels for
3.14 as of this build — if your system default is newer, create the venv
with an explicit 3.12 interpreter).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Requires a nuScenes account + manual download; see the script for details:
./scripts/download_nuscenes_mini.sh /path/to/downloaded/archives

pytest tests/ -v   # 46 tests, all pass without any real data
```

Once nuScenes-mini is downloaded to `configs/data/nuscenes_mini.yaml`'s
`dataroot`:

```bash
python -m roadvision.training.train                              # single GPU
torchrun --nproc_per_node=N -m roadvision.training.train \
  training.distributed.enabled=true                               # multi-GPU DDP

python -m roadvision.eval.evaluate \
  training.checkpoint.resume_path=checkpoints/epoch_0010.pt

python scripts/visualize_bev.py --checkpoint checkpoints/epoch_0010.pt
```

## Repo layout

```
configs/            # Hydra configs: data, model, training
src/roadvision/
  data/              # nuScenes wrapper, projection geometry, BEV GT rasterization, augmentation
  models/            # backbone, depth head, lift-splat pooling, BEV encoder, task heads
  losses/            # per-task losses + uncertainty-weighted multi-task loss
  training/          # train loop (DDP/AMP/checkpointing), W&B tracking
  eval/              # metrics, failure-mode slicing, evaluation entrypoint
scripts/             # dataset download, BEV visualization
tests/               # 46 tests: geometry, BEV pooling, model, losses, training, metrics, failure analysis
docs/                # ARCHITECTURE.md, SYSTEM_DESIGN.md, RESULTS.md
```
