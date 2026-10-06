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

from vla.data.obs_adapter import (
    COMPACT_EE_STATE_DIM,
    ObsAdapterConfig,
    build_vla_inputs,
    compose_state_from_position_buffer,
)
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
        self._position_history: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._adapt(obs, history_mode="reset"), info

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
    def reset_state_history(self, raw_obs: Mapping[str, Any]) -> None:
        """Reset online history from a raw observation without decoding images."""
        compact_state = self._compact_state(raw_obs)
        self._reset_position_history(compact_state)

    def observe_raw_state(self, raw_obs: Mapping[str, Any]) -> None:
        """Append a raw observation to history during a scripted prefix."""
        compact_state = self._compact_state(raw_obs)
        self._append_position_history(compact_state)

    def adapt_current(self, raw_obs: Mapping[str, Any]) -> dict[str, Any]:
        """Adapt an already-observed raw frame without appending it twice."""
        return self._adapt(raw_obs, history_mode="current")

    def _compact_state(self, raw_obs: Mapping[str, Any]) -> torch.Tensor:
        state_parts = []
        for path in self.adapter_cfg.state_keys:
            value: Any = raw_obs
            for part in path.split("/"):
                value = value[part]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Observation state '{path}' is not a tensor")
            state_parts.append(value if value.ndim == 2 else value.unsqueeze(-1))
        state = torch.cat(state_parts, dim=-1).float()
        if self.adapter_cfg.position_history_offsets and state.shape[-1] != COMPACT_EE_STATE_DIM:
            raise ValueError(
                "Position history requires compact 9D state before augmentation, "
                f"got {tuple(state.shape)}"
            )
        return state

    def _reset_position_history(self, compact_state: torch.Tensor) -> None:
        offsets = self.adapter_cfg.position_history_offsets
        if not offsets:
            self._position_history = None
            return
        history_len = offsets[-1] + 1
        self._position_history = compact_state[:, None, :3].repeat(1, history_len, 1)

    def _append_position_history(self, compact_state: torch.Tensor) -> None:
        if not self.adapter_cfg.position_history_offsets:
            return
        if self._position_history is None:
            self._reset_position_history(compact_state)
            return
        self._position_history = torch.cat(
            [self._position_history[:, 1:], compact_state[:, None, :3]], dim=1
        )

    def _adapt(
        self,
        raw_obs: Mapping[str, Any],
        *,
        history_mode: str = "append",
    ) -> dict[str, Any]:
        num_envs = self.env.unwrapped.num_envs
        inputs = build_vla_inputs(
            raw_obs, self.adapter_cfg, task=self.task_prompt, num_envs=num_envs
        )
        compact_state = inputs["state"].float()
        if history_mode == "reset":
            self._reset_position_history(compact_state)
        elif history_mode == "append":
            self._append_position_history(compact_state)
        elif history_mode != "current":
            raise ValueError(f"Unknown history mode: {history_mode}")

        if self.adapter_cfg.position_history_offsets:
            if self._position_history is None:
                raise RuntimeError("Position history has not been initialized")
            inputs["state"] = compose_state_from_position_buffer(
                self._position_history,
                compact_state,
                self.adapter_cfg.position_history_offsets,
            )
        return inputs
