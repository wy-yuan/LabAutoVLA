"""Lightweight logger abstraction.

We use BOTH tensorboard and (optionally) wandb behind a single facade so
the training loop never cares which backend the user configured. Add a
new backend by extending :class:`Logger` — no call-site changes needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

log = logging.getLogger(__name__)


class Logger:
    """Fan out scalars / images / videos to every active backend."""

    def __init__(
        self,
        log_dir: str | Path,
        use_wandb: bool = False,
        wandb_project: str = "labauto-vla",
        wandb_run_name: str | None = None,
        config: Mapping[str, Any] | None = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_url: str | None = None

        # -- Tensorboard (always on — it's free and local) ------------
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(log_dir=str(self.log_dir))
        except ImportError:  # pragma: no cover
            self.tb = None
            log.warning("tensorboard not available; scalar logging disabled")

        # -- Weights & Biases (optional) -----------------------------
        self.wandb = None
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    dir=str(self.log_dir),
                    config=dict(config) if config else None,
                )
                if wandb.run is not None:
                    self.wandb_url = wandb.run.url
            except ImportError:  # pragma: no cover
                log.warning("wandb requested but not installed — skipping")
            except Exception:
                self.wandb = None
                log.exception("wandb requested but failed to initialize; skipping")

    def scalar(self, key: str, value: float, step: int) -> None:
        if self.tb is not None:
            self.tb.add_scalar(key, value, step)
        if self.wandb is not None:
            self.wandb.log({key: value}, step=step)

    def video(self, key: str, path: str | Path, step: int) -> None:
        """Log a video FILE (mp4) — we've already written it to disk."""
        if self.wandb is not None:
            self.wandb.log({key: self.wandb.Video(str(path))}, step=step)
        # Tensorboard wants a tensor; skip for simplicity.

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()
        if self.wandb is not None:
            self.wandb.finish()
