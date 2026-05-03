"""Gymnasium wrapper: Matterix env -> VLA-friendly observations.

Responsibilities
----------------
* Build VLA inputs every step via :func:`build_vla_inputs`, so both the
  training loop and the evaluator can just call ``env.step(action)`` and
  get back a dict the model understands.
* Expose ``env.render()`` that returns the overhead camera frame as a
  ``(H, W, 3)`` uint8 array, so the evaluator can assemble rollout videos
  without reaching into Matterix internals.

This wrapper intentionally does NOT clip / transform actions — whatever
the VLA emits goes straight to the Matterix IK controller. If a model
ever needs output rescaling, do it inside that model's ``predict_action``
rather than here (keeps the wrapper policy-agnostic).
"""

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import torch

from vla.data.obs_adapter import ObsAdapterConfig, build_vla_inputs
from vla.models.smolvla import action_to_env_action


class VLAEnvWrapper(gym.Wrapper):
    """Adapts a Matterix env for a :class:`~vla.models.base_vla.BaseVLA`."""

    def __init__(
        self,
        env: gym.Env,
        adapter_cfg: ObsAdapterConfig,
        task_prompt: str,
        render_camera_key: str = "overhead_camera",
    ):
        super().__init__(env)
        self.adapter_cfg = adapter_cfg
        self.task_prompt = task_prompt
        self.render_camera_key = render_camera_key

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._adapt(obs), info

    def step(self, action):
        # Matterix envs return (obs, reward, terminated, truncated, info).
        action = action_to_env_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._adapt(obs), reward, terminated, truncated, info

    def render(self, mode: str = "rgb_array") -> np.ndarray | None:
        """Grab an RGB frame from the scene camera, no matter what gym mode asks."""
        if self.render_camera_key not in self.env.unwrapped.scene.keys():
            return None
        cam = self.env.unwrapped.scene[self.render_camera_key]
        rgb = cam.data.output["rgb"][..., :3]  # (N, H, W, 3)
        if rgb.dtype.is_floating_point:
            rgb = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        # Return env 0 by default — evaluator handles multi-env stitching.
        return rgb[0].detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _adapt(self, raw_obs: Mapping[str, Any]) -> dict[str, Any]:
        num_envs = self.env.unwrapped.num_envs
        return build_vla_inputs(
            raw_obs, self.adapter_cfg, task=self.task_prompt, num_envs=num_envs
        )
