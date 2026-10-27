"""Rollout + video-recording evaluator used during BC training.

Works against any ``BaseVLA`` and any ``VLAEnvWrapper`` — keeps the
training loop free of simulator-specific code.

Expected call pattern::

    evaluator = RolloutEvaluator(env=wrapped_env, n_episodes=3,
                                 video_dir="runs/eval_videos")
    metrics = evaluator.run(vla, step=global_step)
    # metrics = {"success_rate": 0.66, "episode_length": 412.0, ...}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch

from vla.models.base_vla import BaseVLA

log = logging.getLogger(__name__)


class RolloutEvaluator:
    def __init__(
        self,
        env,
        n_episodes: int = 3,
        max_steps: int = 500,
        video_dir: str | Path | None = None,
        video_fps: int = 30,
    ):
        self.env = env
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.video_dir = Path(video_dir) if video_dir else None
        self.video_fps = video_fps
        if self.video_dir is not None:
            self.video_dir.mkdir(parents=True, exist_ok=True)

    def run(self, policy: BaseVLA, step: int) -> dict[str, float]:
        if self.n_episodes <= 0:
            raise ValueError(f"n_episodes must be positive, got {self.n_episodes}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")

        num_envs = int(getattr(self.env.unwrapped, "num_envs", 1))
        if num_envs <= 0:
            raise ValueError(f"Environment num_envs must be positive, got {num_envs}")

        policy.eval()
        successes: list[float] = []
        lengths: list[int] = []
        print(
            f"[evaluator] starting {self.n_episodes} episodes with num_envs={num_envs}",
            flush=True,
        )

        while len(successes) < self.n_episodes:
            batch_start = len(successes)
            batch_size = min(num_envs, self.n_episodes - batch_start)
            policy.reset()
            obs, _ = self.env.reset()

            active = torch.zeros(num_envs, dtype=torch.bool)
            active[:batch_size] = True
            batch_success = torch.zeros(num_envs, dtype=torch.bool)
            batch_lengths = torch.zeros(num_envs, dtype=torch.long)
            writers: list[Any | None] = [None] * batch_size
            video_paths = [
                self.video_dir / f"rollout_step{step:06d}_ep{batch_start + env_idx}.mp4"
                if self.video_dir is not None
                else None
                for env_idx in range(batch_size)
            ]

            print(
                f"[evaluator] starting batch at episode {batch_start} "
                f"with {batch_size}/{num_envs} active envs",
                flush=True,
            )
            try:
                for t in range(self.max_steps):
                    frame_batch = self._render_batch(num_envs)
                    if frame_batch is not None and self.video_dir is not None:
                        for env_idx in range(batch_size):
                            if not active[env_idx]:
                                continue
                            if writers[env_idx] is None:
                                writers[env_idx] = imageio.get_writer(
                                    video_paths[env_idx],
                                    fps=self.video_fps,
                                    codec="libx264",
                                )
                            writers[env_idx].append_data(
                                np.ascontiguousarray(frame_batch[env_idx])
                            )

                    with torch.inference_mode():
                        action = policy.predict_action(
                            images=obs["images"], state=obs["state"], task=obs["task"]
                        )
                    active_on_action_device = active.to(action.device)
                    if not torch.isfinite(action[active_on_action_device]).all():
                        raise RuntimeError(
                            f"Policy produced non-finite action in batch starting at "
                            f"episode {batch_start}, step {t}"
                        )

                    # Matterix auto-resets individual completed envs. Keep those
                    # reset envs stationary while the rest of the batch finishes.
                    action = self._hold_inactive_envs(action, obs["state"], active)
                    obs, _, terminated, truncated, _ = self.env.step(action)

                    batch_lengths[active] += 1
                    terminated_vec = self._done_vector(terminated, num_envs)
                    truncated_vec = self._done_vector(truncated, num_envs)
                    just_finished = active & (terminated_vec | truncated_vec)
                    batch_success[just_finished] = terminated_vec[just_finished]
                    active[just_finished] = False
                    if not bool(active.any()):
                        break
            finally:
                for env_idx, writer in enumerate(writers):
                    if writer is not None:
                        writer.close()
                        log.info("Saved rollout video: %s", video_paths[env_idx])

            successes.extend(batch_success[:batch_size].float().tolist())
            lengths.extend(batch_lengths[:batch_size].tolist())

        return {
            "success_rate": float(np.mean(successes)) if successes else 0.0,
            "episode_length": float(np.mean(lengths)) if lengths else 0.0,
            "n_episodes": float(len(successes)),
            "num_envs": float(num_envs),
        }

    def _render_batch(self, num_envs: int) -> np.ndarray | None:
        render_all = getattr(self.env, "render_all", None)
        if callable(render_all):
            frames = render_all()
        else:
            frame = self.env.render()
            frames = None if frame is None else np.asarray(frame)[None]
        if frames is None:
            return None
        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[0] != num_envs:
            raise ValueError(
                "Batched render must have shape (num_envs, H, W, C), "
                f"got {frames.shape} for num_envs={num_envs}"
            )
        return frames

    @staticmethod
    def _done_vector(done: Any, num_envs: int) -> torch.Tensor:
        vector = torch.as_tensor(done, dtype=torch.bool).detach().cpu().flatten()
        if vector.numel() == 1 and num_envs > 1:
            vector = vector.expand(num_envs)
        if vector.numel() != num_envs:
            raise ValueError(
                f"Done signal must contain {num_envs} values, got shape {tuple(vector.shape)}"
            )
        return vector

    @staticmethod
    def _hold_inactive_envs(
        action: torch.Tensor,
        state: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        """Replace inactive rows with the current 8D EE pose/gripper command."""
        if bool(active.all()):
            return action
        if action.ndim != 2 or action.shape[-1] != 8:
            raise ValueError(
                "Parallel evaluation requires 8D environment actions to hold completed envs, "
                f"got shape {tuple(action.shape)}"
            )
        current = state[:, -1] if state.ndim == 3 else state
        if current.shape[-1] < 9:
            raise ValueError(
                "Parallel evaluation hold action requires current compact 9D state at the "
                f"start of observation.state, got shape {tuple(current.shape)}"
            )
        current = current.to(device=action.device, dtype=action.dtype)
        gripper = current[:, 7:9].abs().mean(dim=-1, keepdim=True).div(0.04).clamp(0.0, 1.0)
        hold = torch.cat([current[:, :7], gripper], dim=-1)
        result = action.clone()
        result[~active.to(action.device)] = hold[~active.to(action.device)]
        return result
