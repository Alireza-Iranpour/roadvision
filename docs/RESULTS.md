# Results

**A real training run has now been executed** against downloaded nuScenes-mini
(10 scenes, 404 samples, full map-expansion vector maps), on a single CPU-only
development machine. This page reports what was actually measured, including
where the run's scale and a mid-run operational incident limit what can
honestly be concluded — see "What this does and doesn't prove" at the bottom
before reading the numbers as more than they are.

## Train/val split

nuScenes-mini has no official train/val split. One is fixed by scene name in
`configs/data/nuscenes_mini.yaml` (`scene_split`), chosen to put one day scene
and one night(+rain) scene in val so day/night generalization is checkable on
both sides:

- **Train** (8 scenes, 323 samples): scene-0061, 0103, 0553, 0757, 0796, 0916,
  1077, 1100 (includes 2 of the dataset's 3 night scenes)
- **Val** (2 scenes, 81 samples): scene-0655 (day, boston-seaport),
  scene-1094 (night, "after rain", singapore-hollandvillage)

## Baseline: real metrics (full run)

2 epochs over the full 323-sample train split, batch size 2 (646 total
sample-presentations), ResNet-34 backbone, CPU-only, ~26 minutes wall-clock.
Checkpoint: `checkpoints/phase5_baseline/epoch_0001.pt`. This is the only run
in this project that trained for a realistic number of steps — treat it as
the project's actual "does the pipeline learn anything" result, and treat the
smaller runs further down as diagnostic/ablation-budget runs, not as
comparable absolute performance numbers.

| Metric | Value |
|---|---|
| Depth AbsRel | 0.628 |
| Depth SqRel | 9.09 |
| Depth RMSE | 13.54 |
| Depth RMSE-log | 0.571 |
| Depth δ<1.25 | 0.317 |
| Depth δ<1.25² | 0.587 |
| Depth δ<1.25³ | 0.773 |
| Segmentation IoU (drivable_area) | 0.556 |
| Segmentation IoU (lane) | 0.188 |
| Segmentation mean IoU | 0.372 |
| Detection mean AP (simplified proxy — see `metrics.py`) | 0.623 |

These are the numbers of a model trained for ~322 optimizer steps from
ImageNet-pretrained-backbone initialization on 323 real driving-scene samples
— i.e. genuinely under-trained, not a claim of strong absolute perception
quality. AbsRel 0.63 and mean IoU 0.37 are consistent with "the architecture
and loss are wired correctly and something is being learned" (depth δ<1.25
improving from near-chance, lane IoU nonzero despite being the rarer class),
not with a deployable model.

### Per-class detection AP (baseline, full run)

| Class | AP | GT instances in val |
|---|---|---|
| car | 0.214 | 1845 |
| truck | 0.618 | 136 |
| bus | 0.434 | 68 |
| trailer | 0.878 | 10 |
| construction_vehicle | 0.750 | 40 |
| pedestrian | 0.160 | 1021 |
| motorcycle | 0.761 | 20 |
| bicycle | 0.714 | 59 |
| traffic_cone | 0.705 | 190 |
| barrier | 1.000 (vacuous — 0 GT instances in val) | 0 |

**This is the opposite of the naive "rare classes suffer" hypothesis**: the
two most common classes (car, pedestrian) have the *lowest* AP, and classes
with only 10-60 validation instances score highest. The most likely
explanation is AP-estimate noise, not genuine rare-class mastery: with 10-60
GT instances, a handful of lucky/unlucky matches swings the per-class AP by a
large amount, while car's AP (1845 instances) is a much more statistically
stable — and much lower — estimate. **Only car and pedestrian have enough
validation instances (>1000) for their AP numbers to be trusted as anything
more than noise**; the rest should be read as "too few samples to conclude
anything," not as evidence the model handles rare classes well.

## Failure analysis: day/night generalization gap (real, from val predictions)

Per-sample depth AbsRel, sliced by real nuScenes scene attributes
(`eval/failure_analysis.nuscenes_scene_attributes`), on the baseline/full
checkpoint:

| Slice | n | Mean depth AbsRel |
|---|---|---|
| day | 41 | 0.608 |
| night | 40 | 0.653 |
| clear | 41 | 0.608 |
| rain | 40 | 0.653 |

Night is ~7.3% worse than day on depth AbsRel. **Caveat**: val's single
night scene (1094) is also the single rain scene, so "night" and "rain" are
perfectly correlated here, not independent signals — this split cannot tell
night-effect apart from rain-effect, only "night+rain scene vs. day+clear
scene." With one scene on each side of this slice, this is a real but
weak-evidence finding (n=1 scene per condition), consistent with a plausible
day/night domain gap but not a statistically strong one.

## Two diagnosed failure modes and the fixes applied

**1. Detection class imbalance.** Real nuScenes-mini annotation counts
(`sample_annotation` table): car=7619, pedestrian=5040, barrier=2323,
traffic_cone=1378, truck=649, motorcycle=471, bus=410, bicycle=243,
construction_vehicle=196, trailer=60 — a ~127:1 car:trailer ratio. The
original `gaussian_focal_loss` normalized the summed heatmap loss by the
*total* object-center count pooled across all classes, so a rare class's
few centers contributed a tiny fraction of the loss-normalization
denominator relative to its own difficulty. **Fix**: per-class
normalization (`losses/detection_loss.py:gaussian_focal_loss`,
`per_class_normalize=True`, now the default) — each class's loss is
normalized by its own center count before averaging across classes. Verified
at the unit level (`tests/test_losses.py`) with a synthetic 20-centers vs
1-center case showing the rare class gets a comparable per-center gradient
magnitude instead of ~20x smaller.

**2. Day/night generalization gap.** Diagnosed above. **Fix**: stronger
photometric augmentation (`configs/experiment/phase5_cpu_mini_fixed.yaml`:
brightness/contrast jitter 0.2→0.5, saturation 0.2→0.3), intended to make
training exposure/contrast variation closer to the day/night gap the model
will see at eval/deployment time.

### Before/after: NOT a controlled ablation (explained below), but here's what was measured

An attempt to re-run the full 2-epoch/batch=2 training with both fixes
applied hit a real memory-pressure incident on this development machine (see
below) and had to be aborted partway through epoch 2. Rather than report a
broken run, two smaller, *equal-budget* comparisons were run instead — same
48-sample subset of the train split, same random seed, 1 single-sample pass,
baseline code/config vs. fixed code/config:

| Metric | Baseline (48 samples, 1 pass) | Fixed (48 samples, 1 pass) |
|---|---|---|
| Depth AbsRel | 0.814 | 0.834 |
| Segmentation mean IoU | 0.198 | 0.191 |
| Detection mean AP | 0.595 | 0.599 |

Per-class detection AP deltas (fixed − baseline) ranged from **-0.005 to
+0.010** across all 10 classes — i.e. no detectable difference, in either
direction, at this budget.

**Honest interpretation**: 48 samples / 1 gradient pass per weight from a
fresh ImageNet-pretrained backbone is nowhere near enough training for either
fix's effect to manifest in the loss landscape — both configurations are
still extremely close to their shared initialization. This is the expected,
honest result of an equal-budget comparison this small, not evidence the
fixes don't work. The per-class loss normalization fix's actual claim is
verified directly at the unit-test level (gradient magnitude, not downstream
metric); the augmentation fix's actual claim (closing a day/night gap) would
need a multi-epoch run on both sides with real resources to measure, which
this environment could not complete for the reason below.

## Real operational incident: CPU memory pressure during the fixed-config run

~24 minutes into the second epoch of the full 2-epoch/batch=2 "fixed"
training run, the process's CPU progress stalled near zero (process state
`U`, uninterruptible I/O wait) while system swap usage sat at ~18.5GB/19.4GB
and free RAM at ~16MB. The process was killed rather than waited on
indefinitely. A retry at batch_size=1 for the full dataset showed the same
stalling pattern partway through. Root cause was not conclusively isolated —
plausible contributors: (a) this development machine was already running
close to its memory ceiling independent of this project, and (b)
`NuScenesBEVDataset.__getitem__` recomputes LIDAR depth projection and
nuScenes map-expansion polygon queries from scratch on every access rather
than caching them, which is real per-sample CPU/memory cost repeated every
epoch and is a legitimate scaling concern independent of what caused this
specific stall (documented as a system-design consideration in
`docs/SYSTEM_DESIGN.md`). This is reported here rather than hidden because a
training run that silently stalls for 20+ minutes before being noticed is
exactly the kind of operational failure mode a production training pipeline
needs monitoring/alerting for — see `docs/SYSTEM_DESIGN.md`'s monitoring
section.

## Test suite

**53/53 tests passing**, up from 46 at the end of Phase 4 — 7 new tests added
during Phase 5, all regression tests for real bugs found integrating against
real data (not synthetic-only tests this time):

- `tests/test_transforms.py` (4 tests, new file): catches the depth-map/
  augmentation misalignment bug (depth maps were left at raw resolution while
  images were resized+cropped, so `DepthLoss` was comparing predictions for
  an augmented crop against depth values from a different field of view —
  found when the real pipeline crashed with "depth map size (900, 1600) not
  divisible by stride 16").
- `tests/test_losses.py` (+2 tests): verify per-class heatmap loss
  normalization actually equalizes gradient magnitude across common/rare
  classes.
- `tests/test_metrics.py` (+1 test): catches the per-class detection AP bug
  (evaluation was using the class-agnostic `det_target["mask"]` as ground
  truth for every class's AP, so e.g. a car's position could count as a
  correct match for the bicycle head — found while computing per-class AP
  on real predictions and seeing implausible numbers).

Run `pytest tests/ -v` to reproduce.

## What this does and doesn't prove

**Does show**: the full pipeline — real nuScenes data loading, camera/LIDAR
geometry, BEV ground-truth rasterization against real map-expansion vector
maps, the multi-task LSS-style forward pass, all three losses, the
uncertainty-weighted combination, checkpointing, and the evaluation/
failure-analysis tooling — runs correctly end-to-end against real data and
produces metrics that move in sensible directions during training (not NaN,
not frozen, not degenerate). It also shows the project's "diagnose failure
mode → implement targeted fix → verify" process works: both fixes are
grounded in real measurements (the 127:1 annotation imbalance, the real
day/night AbsRel gap), not speculative guesses.

**Does not show**: that either fix measurably improves the trained model,
because no equal-budget run large enough to detect that effect could be
completed in this environment (blocked by the memory incident above, not by
a flaw in the fixes' design). It also does not show strong absolute model
performance — 323-646 training samples and a few hundred optimizer steps on
CPU is a correctness/pipeline-validation scale, not a scale anyone would
expect real accuracy from. A real verification of these fixes' effect needs
a GPU (removing the CPU wall-clock/memory constraint entirely) and a
multi-epoch run on both configurations under equal conditions.
