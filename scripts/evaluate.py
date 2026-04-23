# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hydra entry point for simulation-only VLA evaluation.

This script loads a VLA policy, builds the Matterix simulation environment,
and runs rollout evaluation without constructing datasets, optimizers, or any
training loop.

Examples
--------
Evaluate the default pretrained SmolVLA on the default pipetting task:
    python scripts/evaluate.py

Override rollout length or episode count:
    python scripts/evaluate.py eval.n_episodes=3 eval.max_steps=300

Evaluate another model/task config:
    python scripts/evaluate.py model=smolvla task=pipetting
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

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


def _launch_sim_app():
    """Launch Isaac Sim before importing Matterix/Isaac Lab task code."""
    global _app_launcher, simulation_app
    if simulation_app is not None:
        return simulation_app

    print("[evaluate] Launching Isaac Sim for evaluation...", flush=True)
    from isaaclab.app import AppLauncher

    _app_launcher = AppLauncher(headless=True, enable_cameras=True, livestream=2)
    simulation_app = _app_launcher.app
    print("[evaluate] Isaac Sim launcher returned control.", flush=True)
    return simulation_app


def _build_env(cfg: DictConfig):
    """Build and wrap the simulation env used by the rollout evaluator."""
    import gymnasium as gym
    import matterix_tasks  # noqa: F401 - registers envs
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from vla.data.obs_adapter import ObsAdapterConfig
    from vla.envs.vla_env_wrapper import VLAEnvWrapper

    adapter_cfg = ObsAdapterConfig(**OmegaConf.to_container(cfg.task.adapter, resolve=True))
    env_cfg = parse_env_cfg(cfg.task.id, device=cfg.device, num_envs=1)

    log = logging.getLogger(__name__)
    log.info("Building evaluator env id=%s device=%s num_envs=1", cfg.task.id, cfg.device)
    env = gym.make(cfg.task.id, cfg=env_cfg).unwrapped
    return VLAEnvWrapper(env, adapter_cfg=adapter_cfg, task_prompt=cfg.task.prompt)


def _infer_action_dim(env: Any) -> int:
    """Infer single-env action dimension from a Gym/Matterix env."""
    action_space = getattr(env.unwrapped, "single_action_space", None)
    if action_space is None:
        action_space = env.action_space

    if not hasattr(action_space, "shape") or len(action_space.shape) == 0:
        raise ValueError(f"Cannot infer action_dim from action space: {action_space!r}")
    return int(action_space.shape[-1])


def _build_policy(cfg: DictConfig, obs: dict[str, Any], action_dim: int):
    """Instantiate the configured VLA policy from the evaluation observation schema."""
    from vla.models import build_vla

    state_dim = int(obs["state"].shape[-1])
    image_keys = list(obs["images"].keys())
    model_kwargs = dict(OmegaConf.to_container(cfg.model.get("kwargs", {}), resolve=True))
    model_kwargs["device"] = cfg.device

    log = logging.getLogger(__name__)
    log.info(
        "Building VLA model name=%s state_dim=%d action_dim=%d image_keys=%s",
        cfg.model.name,
        state_dim,
        action_dim,
        image_keys,
    )
    vla = build_vla(
        cfg.model.name,
        action_dim=action_dim,
        state_dim=state_dim,
        image_keys=image_keys,
        **model_kwargs,
    )
    vla.to(torch.device(cfg.device))
    return vla


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    _configure_logging()
    log = logging.getLogger(__name__)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    log.info("Evaluation config:\n%s", OmegaConf.to_yaml(cfg))

    _launch_sim_app()
    env = None
    try:
        env = _build_env(cfg)
        obs, _ = env.reset()
        action_dim = _infer_action_dim(env)
        vla = _build_policy(cfg, obs, action_dim)

        from vla.training.evaluator import RolloutEvaluator

        evaluator = RolloutEvaluator(
            env=env,
            n_episodes=cfg.eval.n_episodes,
            max_steps=cfg.eval.max_steps,
            video_dir=Path(cfg.output_dir) / "eval_videos",
            video_fps=cfg.dataset.fps,
        )
        metrics = evaluator.run(vla, step=0)
        log.info("Evaluation metrics: %s", metrics)
        print(f"[evaluate] metrics: {metrics}", flush=True)
    finally:
        if env is not None:
            env.close()
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    # Ensure relative paths stay rooted at the repository, matching scripts/train.py.
    os.chdir(_REPO_ROOT)
    main()
