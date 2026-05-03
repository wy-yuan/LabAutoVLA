"""Rollout + video-recording evaluator used after every BC epoch.

Works against any ``BaseVLA`` and any ``VLAEnvWrapper`` — keeps the
training loop free of simulator-specific code.

Expected call pattern::

    evaluator = RolloutEvaluator(env=wrapped_env, n_episodes=3,
                                 video_dir="runs/eval_videos")
    metrics = evaluator.run(vla, step=epoch)
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
        policy.eval()
        successes: list[float] = []
        lengths: list[int] = []
        print("[evaluator] starting running policy !!!", flush=True)
        for ep in range(self.n_episodes):
            policy.reset()
            obs, info = self.env.reset()

            frames: list[np.ndarray] = []
            prev_frame: np.ndarray | None = None
            success = False
            t = 0
            print(f"[evaluator] starting episode {ep}", flush=True)
            for t in range(self.max_steps):
                frame = self.env.render()
                print(f"[evaluator] self.env.render step {t}", flush=True)
                if frame is not None:
                    # makes a NumPy array whose memory is laid out 
                    # continuously in normal row-major order
                    frame = np.ascontiguousarray(frame)
                    frames.append(frame)
                else:
                    print("[evaluator] render returned None", flush=True)
                # print(obs["task"], flush=True)
                with torch.inference_mode():
                    action = policy.predict_action(
                        images=obs["images"], state=obs["state"], task=obs["task"]
                    )
                # print(
                #     f"[evaluator] action predicted  "
                #     f"pos={action[..., :3].tolist()}  "
                #     f"has_nan={bool(torch.isnan(action).any())}  "
                #     f"has_inf={bool(torch.isinf(action).any())}",
                #     flush=True,
                # )
                # Ensure all PyTorch CUDA work finishes before Isaac Sim's
                # render pass inside env.step (different internal streams).
                # if action.is_cuda:
                #     torch.cuda.synchronize(action.device)
                obs, reward, terminated, truncated, info = self.env.step(action)
                if bool(torch.as_tensor(terminated).any()):
                    # Matterix termination == workflow success in most tasks.
                    success = True
                    break
                if bool(torch.as_tensor(truncated).any()):
                    break

            successes.append(float(success))
            lengths.append(t + 1)

            if self.video_dir is not None and frames:
                out = self.video_dir / f"rollout_step{step:06d}_ep{ep}.mp4"
                imageio.mimsave(out, frames, fps=self.video_fps, codec="libx264")
                log.info("Saved rollout video: %s", out)

        return {
            "success_rate": float(np.mean(successes)) if successes else 0.0,
            "episode_length": float(np.mean(lengths)) if lengths else 0.0,
            "n_episodes": float(len(successes)),
        }
