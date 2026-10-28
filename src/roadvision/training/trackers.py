"""Thin W&B wrapper.

A training run must never crash because of a transient W&B/network issue
or a missing API key in a dev/CI environment, so every call here catches
and logs instead of propagating -- this is a "best-effort telemetry"
component, not part of the training correctness path.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class WandbTracker:
    def __init__(
        self,
        project: str,
        config: dict[str, Any],
        mode: str = "online",
        entity: str | None = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._run = None
        if not enabled:
            return

        try:
            import wandb

            self._run = wandb.init(project=project, entity=entity, config=config, mode=mode)
        except Exception:
            logger.warning("W&B init failed; continuing without experiment tracking.", exc_info=True)
            self.enabled = False

    def log_scalars(self, scalars: dict[str, float], step: int) -> None:
        if not self.enabled or self._run is None:
            return
        try:
            self._run.log(scalars, step=step)
        except Exception:
            logger.warning("W&B scalar logging failed; continuing.", exc_info=True)

    def log_images(self, images: dict[str, Any], step: int) -> None:
        if not self.enabled or self._run is None:
            return
        try:
            import wandb

            self._run.log({k: wandb.Image(v) for k, v in images.items()}, step=step)
        except Exception:
            logger.warning("W&B image logging failed; continuing.", exc_info=True)

    def finish(self) -> None:
        if self.enabled and self._run is not None:
            try:
                self._run.finish()
            except Exception:
                logger.warning("W&B finish() failed; continuing.", exc_info=True)


class NullTracker:
    """No-op tracker, used when `logging.backend` is disabled entirely."""

    def log_scalars(self, scalars: dict[str, float], step: int) -> None:
        pass

    def log_images(self, images: dict[str, Any], step: int) -> None:
        pass

    def finish(self) -> None:
        pass
