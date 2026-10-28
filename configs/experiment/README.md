# Experiment presets

Experiment configs go here as Hydra config groups that override
`config.yaml` defaults, e.g. `configs/experiment/depth_only.yaml` to
ablate the detection/segmentation losses, or
`configs/experiment/full_nuscenes.yaml` to switch `data` to the full
dataset. None exist yet -- added alongside the training pipeline in a
later phase.
