# System Design

This document separates what's actually implemented in this repo from what
is discussed as a future-scale direction. Everything under "Implemented"
cites a real file; everything under "At scale" or "Future direction" is
design reasoning, not code in this repo.

## Data pipeline

**Implemented:** `NuScenesBEVDataset` (`src/roadvision/data/nuscenes_dataset.py`)
loads nuScenes-mini directly off local disk through `nuscenes-devkit`,
decoding images and projecting LIDAR to per-camera depth on every `__getitem__`
call, and `NuScenesBEVAnnotations` (`src/roadvision/data/bev_rasterize.py`)
rasterizes 3D boxes and map polygons into BEV targets per sample, also on
every call. This is the right amount of engineering for a 10-scene mini
dataset running on one machine — anything more would be premature.

**At full-nuScenes scale, three things change:**

1. **Re-decoding cost.** Image JPEG decode + LIDAR point-cloud projection
   per sample, per epoch, is wasted work: both are deterministic given the
   sample token and the (fixed) BEV/depth-bin config. The fix is a
   preprocessing pass that writes precomputed depth maps and BEV
   rasterization targets to disk once (e.g. as compressed `.npz` or a
   columnar format), keyed by sample token + a hash of the grid config so a
   config change invalidates the right cache entries instead of silently
   serving stale targets.

   This isn't theoretical: during Phase 5's real-data training run
   (`docs/RESULTS.md`), a CPU-only training process stalled for 20+ minutes
   with system swap at ~18.5GB/19.4GB and near-zero further CPU progress,
   partway through a second epoch. The stall's root cause wasn't
   conclusively isolated (the development machine may have already been
   near its memory ceiling independent of this project), but re-doing
   nuScenes map-expansion polygon queries and LIDAR projection from scratch
   on every `__getitem__` call, every epoch, is real repeated CPU/memory
   cost that this precomputation fix would remove regardless — and a
   training run that silently stalls rather than erroring is exactly the
   failure mode the monitoring section below is for.
2. **Storage/IO at scale.** Full nuScenes is ~300GB of images + LIDAR.
   Random-access `__getitem__` over individual files on a shared filesystem
   doesn't scale past a few GPUs' worth of data-loader workers before IO
   becomes the bottleneck. The standard fix is sharding into sequential-read
   formats (WebDataset tar shards, or Parquet with embedded references to
   object storage) so each worker streams a contiguous shard instead of
   issuing thousands of small random reads — this also makes the dataset
   trivially distributable across a training cluster without a shared POSIX
   filesystem.
3. **Horizontal flip augmentation is currently disabled** (`horizontal_flip_prob: 0.0`,
   enforced in `AugmentationConfig`) specifically because a correct flip
   needs to mirror the BEV targets' y-axis *and* the camera extrinsics in
   lockstep — flipping images without flipping the geometry would silently
   corrupt the lift-splat alignment. This is scoped as real follow-up work,
   not a shortcut: implementing it is what unlocks ~2x effective training
   data for free, which matters more at full scale where training time
   dominates, than on mini where dataset size isn't the bottleneck.

## Distributed training

**Implemented:** `src/roadvision/training/engine.py` reads `RANK`/`WORLD_SIZE`/
`LOCAL_RANK` the way `torchrun` sets them up and wraps the model in
`DistributedDataParallel` when `training.distributed.enabled=true`
(`configs/training/default.yaml`, off by default for single-GPU use). AMP
(`torch.autocast` + `GradScaler`) and gradient accumulation
(`grad_accumulation_steps`) are implemented and independent of the
distributed path, so they apply whether or not DDP is on.

One correctness detail worth calling out explicitly: `MultiTaskUncertaintyLoss`
(`src/roadvision/losses/multitask_loss.py`) owns learned parameters
(`log_vars`), and those need the same gradient all-reduce under DDP as
every other model weight. This was initially missed — `multitask_loss` is
a separate `nn.Module` from `model`, and only `model` was passed through
`engine.wrap_model_for_distributed`, so each rank would have learned
different per-task weights that silently diverged at multi-GPU scale while
single-GPU training looked completely fine. `engine.wrap_model_for_distributed`
is now called on `multitask_loss` too (`training/train.py`), and checkpoint
save/load unwrap it the same way they already unwrapped `model`. Flagged
here anyway because it's exactly the kind of bug that passes single-GPU
testing and breaks multi-GPU training in a way that's hard to notice
(training "works", loss curves just don't match between runs) — worth
double-checking on an actual multi-GPU run before trusting it fully.

**At full scale, single-node DDP stops being enough:**

- **Multi-node training** needs `torchrun`'s multi-node launch (or Ray
  Train, which adds fault tolerance and elastic scaling on top) — the
  current `setup_distributed()` already reads the standard env vars
  `torchrun` provides, so the same entrypoint extends to multi-node without
  code changes, only launch configuration.
- **Gradient sync overhead** grows with world size and model size. For a
  backbone larger than ResNet-34 (e.g. if this moved to a ViT-based
  backbone for a production system), gradient checkpointing trades compute
  for activation memory, and `DistributedDataParallel`'s
  `gradient_as_bucket_view` / static-graph optimizations reduce sync
  overhead — neither is needed at current model size but both are standard
  levers once backbone size grows.
- **Data loader worker scaling** needs to track GPU count: `num_workers: 8`
  (per-process) in `configs/data/nuscenes_mini.yaml` is sized for one GPU
  on one machine; a multi-GPU node needs this divided across the processes
  sharing that node's CPU, not applied per-process unchanged.

## Experiment tracking and reproducibility

**Implemented:** Hydra composes `configs/{data,model,training}/*.yaml` into
one resolved config per run (`configs/config.yaml`), and `WandbTracker`
(`src/roadvision/training/trackers.py`) logs that resolved config at run
start plus per-step scalar losses — including the learned per-task
uncertainty weights, which are a genuinely useful training-health signal,
not just loss curves. W&B failures (no API key, no network) are caught and
degrade to a no-op rather than crashing training (`logging.backend: wandb`
vs `none` toggles it off entirely via `NullTracker`).

Because the dataset/preprocessing config (BEV grid, depth bins, image size)
and the model config are composed into the *same* resolved config object
and logged together, a W&B run fully specifies both what model ran and what
geometric assumptions it was trained under — this matters because changing
the BEV grid resolution changes what a checkpoint's weights mean, and a
model/config mismatch would silently produce wrong (not just worse)
predictions. **Not yet implemented**: dataset *version* pinning (e.g. a
hash of the actual nuScenes map/annotation version used) isn't currently
part of the logged config, which would matter the moment the dataset
itself gets updated upstream.

## Serving and inference

**Not implemented** (out of scope for this build pass by design — see
`docs/RESULTS.md` and the project README for what phases were and weren't
done), but worth reasoning about concretely rather than hand-waving:

A production onboard perception stack at ~10Hz has a ~100ms budget for this
model's full forward pass. Rough breakdown of where that goes, by pipeline
stage (`docs/ARCHITECTURE.md` has the full shape table):

- **Image preprocessing** (resize/crop/normalize, 6 cameras): small and
  parallelizable, not the bottleneck.
- **Backbone forward**, 6x `(3,256,704)` through ResNet-34 truncated at
  `layer3`: the dominant FLOP cost, but also the most standard to optimize
  (TensorRT has mature ResNet kernels, this is not a novel op).
- **Lift-splat pooling** (`bev_pool.py`): this is the part I'd actually
  flag as the realistic bottleneck and engineering risk, for two reasons.
  First, the current implementation is a Python `for b in range(batch_size)`
  loop over per-sample `index_add_` calls — correctness-first as documented
  in the module's own docstring, but that loop doesn't parallelize across
  batch on GPU the way a single batched op would; the standard LSS
  "cumsum-trick" (sort frustum points by BEV cell index, cumsum the
  features, then take boundary differences) avoids both the per-sample
  loop and materializing the full per-point feature tensor, and would be
  the first thing to implement before deploying this for latency-sensitive
  use, not a nice-to-have.
  Second, **`scatter_add`/`index_add_` is a real ONNX/TensorRT export
  risk**, not a hypothetical one: dynamic scatter ops with data-dependent
  indices are exactly the category of op that ONNX export and TensorRT's
  graph optimizer have historically handled poorly or not at all (support
  has improved over time but needs to be verified against whatever
  TensorRT version a given deployment targets — I'm flagging this as an
  open risk to validate, not asserting it's solved). A fallback path if
  export fails is a custom CUDA/TensorRT plugin implementing the voxel-pool
  op directly, which is what production LSS-family systems typically ship.
- **Task heads**: cheap relative to the backbone and pooling step.

**Export path**: an ONNX export entrypoint was explicitly scoped out of
this build pass (see `docs/RESULTS.md`) specifically because validating it
properly means confirming the scatter-add op traces/exports correctly for
a target TensorRT version, which isn't meaningfully testable without a real
deployment target — shipping an untested export script would overstate
what's actually verified here. **Quantization** (INT8 via TensorRT's
post-training calibration) is a reasonable next optimization after export
is working, likely applied to the backbone first since it's the FLOP-heavy,
well-understood-to-quantize part, with the lift-splat and heads evaluated
separately for accuracy sensitivity before quantizing those too.

## Monitoring and reliability in production

**Not implemented** (design-only), but directly connects to what Phase 4
actually built:

- **Input distribution drift**: the depth-distribution head's output is a
  genuine probability distribution over 41 bins per pixel (`depth.py`), so
  its *entropy* is a cheap, always-available signal — a sustained rise in
  average predicted-depth entropy across a fleet (without a corresponding
  change in true scene geometry) is a reasonable proxy for "the model is
  seeing something further from training distribution than usual" (new
  weather, new camera hardware, a miscalibrated rig) and is cheaper to
  monitor continuously than running full evaluation metrics on every frame.
  Detection heatmap confidence calibration (are predicted-positive cells
  actually positive at the predicted rate, tracked against whatever ground
  truth becomes available after the fact) is the equivalent signal for the
  detection head.
- **Shadow deployment**: before replacing a production model, run the
  candidate alongside it on live (or recently collected) data without
  acting on its outputs, and compare per-task metrics — this is exactly
  what `src/roadvision/eval/evaluate.py` + `failure_analysis.py` already
  compute offline against a static validation set; the production version
  of this is the same metric/slicing code pointed at a continuously
  refreshed sample of fleet data instead of a fixed val split.
- **Continuous failure-mode monitoring**: `failure_analysis.py`'s
  `slice_metrics_by_attribute` and `worst_k_samples` (built in Phase 4,
  tested in `tests/test_failure_analysis.py`) are generic over any
  per-sample attribute dict — in production that attribute dict would be
  populated from real metadata (time of day, weather, geographic region,
  sensor rig ID) rather than the `nuscenes_scene_attributes` adapter stub
  this repo currently has for nuScenes' own scene descriptions. Wiring that
  same slicing logic into a recurring job over newly collected fleet data
  is how you catch a model quietly degrading on a specific scenario (e.g.
  tunnel entrances, or a new sensor hardware revision) before it shows up
  as an aggregate metric regression, since aggregate metrics are exactly
  what a narrow, rare-but-severe failure mode would get averaged away in.
