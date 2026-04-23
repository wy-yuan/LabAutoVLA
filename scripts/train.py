# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hydra entry point for VLA training.

Examples
--------
Behavior cloning on SmolVLA:
    python scripts/train.py mode=bc model=smolvla task=pipetting

RL fine-tune (placeholder — not yet implemented):
    python scripts/train.py mode=rl model=smolvla task=pipetting

Override anything from the CLI:
    python scripts/train.py mode.batch_size=32 mode.optim.lr=5e-5

See configs/train.yaml for all defaults.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Make `import vla` work when running from the repo root without installing.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_app_launcher = None
simulation_app = None


def _configure_logging() -> None:
    """Configure console logging so Unicode config values do not crash on Windows."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    # Keep our app logs visible while suppressing noisy dependency debug logs.
    # logging.getLogger("filelock").setLevel(logging.WARNING)
    # logging.getLogger("fsspec").setLevel(logging.WARNING)
    # logging.getLogger("fsspec.local").setLevel(logging.WARNING)


def _maybe_launch_sim_app(cfg: DictConfig):
    """Launch Isaac Sim before importing training code when sim eval is enabled."""
    global _app_launcher, simulation_app
    if simulation_app is not None:
        return simulation_app

    if cfg.mode.name == "bc" and bool(cfg.mode.get("sim_eval", True)):
        print("[train] Launching Isaac Sim for evaluator...")
        from isaaclab.app import AppLauncher

        _app_launcher = AppLauncher(headless=True, enable_cameras=True, livestream=2)
        
        simulation_app = _app_launcher.app
        print("[train] Isaac Sim launcher returned control.")

    return simulation_app


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    _configure_logging()
    log = logging.getLogger(__name__)

    log.info("Launching training entrypoint")
    _maybe_launch_sim_app(cfg)
    log.info("Training launcher ready; mode=%s sim_eval=%s", cfg.mode.name, cfg.mode.get("sim_eval", None))

    try:
        mode_name = cfg.mode.name
        if mode_name == "bc":
            log.info("Importing BC trainer")
            from vla.training.bc_train import run_bc
            log.info("Starting BC trainer")
            run_bc(cfg)
        elif mode_name == "rl":
            log.info("Importing RL trainer")
            from vla.training.rl_finetune import run_rl
            log.info("Starting RL trainer")
            run_rl(cfg)
        else:
            raise ValueError(f"Unknown training mode '{mode_name}' (expected 'bc' or 'rl')")
    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()
