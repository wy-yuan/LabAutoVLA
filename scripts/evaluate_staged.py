# Copyright (c) 2022-2026, The Matterix Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a VLA after a scripted prefix, without resetting at the handoff.

Examples:
    isaacpython scripts/evaluate_staged.py handoff.after_stage=picking
    isaacpython scripts/evaluate_staged.py handoff.after_stage=source
    isaacpython scripts/evaluate_staged.py handoff.after_action=5 eval.n_episodes=3

Uses the same headless launcher, checkpoint loading, and normalization as evaluate.py.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys

import hydra
import imageio
import numpy as np
from omegaconf import DictConfig, OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reserve the project package before Kit adds another top-level `data` package.
import data  # noqa: F401
from scripts import evaluate
from vla.training.staged_evaluator import resolve_handoff, run_staged_episode


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate_staged")
def main(cfg: DictConfig) -> None:
    evaluate._configure_logging()
    log = logging.getLogger(__name__)
    if cfg.eval.n_episodes <= 0 or cfg.eval.max_steps <= 0 or cfg.handoff.max_steps < 0:
        raise ValueError("Episode/policy budgets must be positive; scripted budget must be nonnegative")
    output_dir = Path(cfg.output_dir)
    video_dir = output_dir / "eval_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    log.info("Staged evaluation config:\n%s", OmegaConf.to_yaml(cfg))

    env = None
    try:
        evaluate._launch_sim_app()
        # Import simulator-dependent modules only after AppLauncher has started.
        from data.workflow_executor import create_workflow_executor

        env = evaluate._build_env(cfg)
        obs, _ = env.reset()
        policy = evaluate._build_policy(cfg, obs, evaluate._infer_action_dim(env))
        evaluate._ensure_action_normalization(cfg, policy)
        # Simulator initialization can replace Python's logging configuration.
        evaluate._configure_logging()
        executor = create_workflow_executor(
            env.unwrapped,
            env.unwrapped.cfg,
            preferred_workflow_name=cfg.task.workflow.name,
            suppress_output=True,
        )
        completed_actions = resolve_handoff(
            executor, cfg.handoff.after_stage, cfg.handoff.after_action
        )
        sequence = [type(action).__name__ for action in executor.state_machine.actions]
        log.info("Primitive sequence (zero-based): %s", list(enumerate(sequence)))
        log.info("Model takes over after %d completed primitives", completed_actions)
        # Match the actual control rate rather than the training video metadata.
        video_fps = 1.0 / env.unwrapped.step_dt
        results = []
        for episode in range(cfg.eval.n_episodes):
            video_path = video_dir / f"rollout_ep{episode:03d}.mp4"
            frame_count = 0
            with imageio.get_writer(video_path, fps=video_fps, codec="libx264") as writer:
                def record_frame() -> None:
                    nonlocal frame_count
                    frame = env.render()
                    if frame is None:
                        raise RuntimeError("Overhead camera returned no frame during staged evaluation")
                    writer.append_data(np.ascontiguousarray(frame))
                    frame_count += 1

                result = run_staged_episode(
                    env, policy, executor,
                    completed_actions=completed_actions,
                    scripted_max_steps=cfg.handoff.max_steps,
                    max_steps=cfg.eval.max_steps,
                    record_frame=record_frame,
                )
            result.update(episode=episode, video=str(video_path), video_frames=frame_count)
            results.append(result)
            report = {
                "workflow": executor.workflow_name,
                "completed_actions": completed_actions,
                "primitive_sequence": sequence,
                "video_fps": video_fps,
                "handoff_rate": sum(r["handoff_reached"] for r in results) / len(results),
                "episodes": results,
            }
            (output_dir / "staged_metrics.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            log.info(
                "Episode %d: handoff=%s scripted_steps=%d policy_steps=%d reason=%s video=%s",
                episode, result["handoff_reached"], result["scripted_steps"],
                result["policy_steps"], result["stop_reason"], video_path,
            )
    except Exception:
        # Some Kit versions exit inside close(), before Hydra can print the error.
        log.exception("Staged evaluation failed")
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            if evaluate.simulation_app is not None:
                evaluate.simulation_app.close()


if __name__ == "__main__":
    os.chdir(_REPO_ROOT)
    main()
