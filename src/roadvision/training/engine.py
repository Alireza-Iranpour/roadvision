"""Train/val loop: DDP, AMP, gradient accumulation, checkpointing.

`is_distributed`/`rank`/`world_size` are read from the environment the way
`torchrun` sets it up, not re-derived here, so the exact same `train.py`
entrypoint runs unmodified under `python train.py` (single process) or
`torchrun --nproc_per_node=N train.py distributed.enabled=true`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader


@dataclass
class DistributedContext:
    enabled: bool
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def setup_distributed(enabled: bool, backend: str = "nccl") -> DistributedContext:
    if not enabled:
        return DistributedContext(enabled=False)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return DistributedContext(enabled=True, rank=rank, world_size=world_size, local_rank=local_rank)


def wrap_model_for_distributed(model: nn.Module, ctx: DistributedContext) -> nn.Module:
    """Wrap any module with learnable parameters that takes part in the
    training forward/backward pass -- this includes `MultiTaskUncertaintyLoss`,
    not just `RoadVisionNet` itself. Its `log_vars` are optimized jointly with
    the model weights, so under DDP they need the same gradient all-reduce or
    each rank's uncertainty weights silently drift apart on different local
    batches instead of reflecting a consistent global estimate.
    """
    if not ctx.enabled:
        return model
    model = model.to(ctx.local_rank)
    return DistributedDataParallel(model, device_ids=[ctx.local_rank])


@dataclass
class EngineConfig:
    epochs: int
    grad_accumulation_steps: int
    amp: bool
    checkpoint_dir: str
    save_every_n_epochs: int
    keep_last_n: int
    log_every_n_steps: int


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0


ComputeLosses = Callable[[dict[str, Any], nn.Module], dict[str, torch.Tensor]]
"""Builds the multitask-loss dict (see `MultiTaskUncertaintyLoss.forward`)
from a raw batch and the (possibly-DDP-wrapped) model -- kept as an
injected function rather than hardcoded here so `engine.py` doesn't need
to know about `RoadVisionNet`'s specific batch/output schema."""


def train_one_epoch(
    model: nn.Module,
    multitask_loss: nn.Module,
    dataloader: DataLoader,
    compute_losses: ComputeLosses,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: EngineConfig,
    state: TrainState,
    tracker: Any,
    ctx: DistributedContext,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(dataloader):
        batch = _move_to_device(batch, device)

        with torch.autocast(device_type=device.type, enabled=config.amp):
            task_losses = compute_losses(batch, model)
            losses = multitask_loss(task_losses)
            loss = losses["total"] / config.grad_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % config.grad_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            state.global_step += 1

            if ctx.is_main_process and state.global_step % config.log_every_n_steps == 0:
                tracker.log_scalars(
                    {k: v.item() for k, v in losses.items() if k != "total"} | {"total": losses["total"].item()},
                    step=state.global_step,
                )

        total_loss += losses["total"].item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    multitask_loss: nn.Module,
    dataloader: DataLoader,
    compute_losses: ComputeLosses,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for batch in dataloader:
        batch = _move_to_device(batch, device)
        task_losses = compute_losses(batch, model)
        losses = multitask_loss(task_losses)
        total_loss += losses["total"].item()
        num_batches += 1
    return total_loss / max(num_batches, 1)


def _move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    multitask_loss: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    state: TrainState,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unwrapped_model = model.module if isinstance(model, DistributedDataParallel) else model
    unwrapped_loss = multitask_loss.module if isinstance(multitask_loss, DistributedDataParallel) else multitask_loss
    torch.save(
        {
            "model": unwrapped_model.state_dict(),
            "multitask_loss": unwrapped_loss.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": state.epoch,
            "global_step": state.global_step,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    multitask_loss: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
) -> TrainState:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    unwrapped_model = model.module if isinstance(model, DistributedDataParallel) else model
    unwrapped_model.load_state_dict(checkpoint["model"])
    unwrapped_loss = multitask_loss.module if isinstance(multitask_loss, DistributedDataParallel) else multitask_loss
    unwrapped_loss.load_state_dict(checkpoint["multitask_loss"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint["scheduler"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return TrainState(epoch=checkpoint["epoch"], global_step=checkpoint["global_step"])


def prune_old_checkpoints(checkpoint_dir: Path, keep_last_n: int) -> None:
    checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime)
    for stale in checkpoints[:-keep_last_n] if keep_last_n > 0 else []:
        stale.unlink()
