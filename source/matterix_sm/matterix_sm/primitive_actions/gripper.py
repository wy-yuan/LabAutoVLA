# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gripper actions - open and close gripper while holding pose."""

from __future__ import annotations

from dataclasses import MISSING
from typing import ClassVar

import torch

from .._compat import configclass
from ..primitive_action import PrimitiveAction, PrimitiveActionCfg
from ..robot_action_spaces import ActionSpaceInfo
from ..scene_data import SceneData


@configclass
class GripperActionCfg(PrimitiveActionCfg):
    """Configuration for gripper actions (open/close).

    Inherits base fields (assets, timeout) from PrimitiveActionCfg.

    Attributes:
        target_value: Logical gripper command value. Positive opens, negative closes. REQUIRED.
        duration: Time (in seconds) to hold the command. Default: 1.0 second.
        interpolation_duration: Time (in seconds) to ramp from current gripper state to target.
        command_mode: "binary" for sign-based actions or "position" for joint-position targets.
        closed_position: Physical gripper position treated as fully closed.
        open_position: Physical gripper position treated as fully open.
    """

    target_value: float = MISSING
    duration: float = 2.0
    interpolation_duration: float = 0.0
    command_mode: str | None = None
    closed_position: float | None = None
    open_position: float | None = None


@configclass
class OpenGripperCfg(GripperActionCfg):
    """Configuration for OpenGripper action."""

    target_value: float = 1.0


@configclass
class CloseGripperCfg(GripperActionCfg):
    """Configuration for CloseGripper action."""

    target_value: float = -1.0


class GripperAction(PrimitiveAction):
    """Hold current EE pose and apply a gripper command for a fixed duration."""

    cfg_type: ClassVar[type] = GripperActionCfg
    _BINARY_EPS: ClassVar[float] = 1e-6

    def __init__(
        self,
        agent_assets: str | list[str],
        target_value: float,
        duration: float,
        timeout: float,
        interpolation_duration: float = 0.0,
        command_mode: str | None = None,
        closed_position: float | None = None,
        open_position: float | None = None,
        action_space_info: ActionSpaceInfo | None = None,
    ):
        """
        Args:
            agent_assets: Name(s) of articulated asset(s) acting as agents.
            target_value: Logical gripper command value. Positive opens, negative closes.
            duration: Time (in seconds) to maintain the command.
            timeout: Max time (in seconds) before timeout.
            interpolation_duration: Time in seconds to ramp the command from current to target.
            command_mode: Optional override for gripper command mode ("binary" or "position").
            closed_position: Physical gripper observation value treated as fully closed.
            open_position: Physical gripper observation value treated as fully open.
            action_space_info: Optional action space metadata for mask creation.
        """
        super().__init__(agent_assets, timeout, action_space_info)
        self.target_value = float(target_value)
        self.duration = duration
        self.interpolation_duration = interpolation_duration

        if self.action_space_info is None:
            raise ValueError(
                "GripperAction requires action_space_info to determine gripper indices. "
                "Pass action_space_info parameter when creating the action."
            )

        self.command_mode = command_mode or getattr(self.action_space_info, "gripper_command_mode", "binary")
        if self.command_mode not in ("binary", "position"):
            raise ValueError(f"Unsupported gripper command_mode: {self.command_mode}")

        self.closed_position = self._resolve_position(
            closed_position, "gripper_closed_position", default=0.0
        )
        self.open_position = self._resolve_position(
            open_position, "gripper_open_position", default=0.04
        )
        self._gripper_indices = list(self.action_space_info.gripper_indices or [])

        self._action_dim_mask = None
        self._action_tensor = None
        self._interp_start_values = None
        self._interp_initialized = None

    def _resolve_position(self, value: float | None, attr_name: str, default: float) -> float:
        """Resolve an explicit position override or the action-space default."""
        if value is not None:
            return float(value)
        return float(getattr(self.action_space_info, attr_name, default))

    def set_execution_params(self, num_envs: int, device: str | torch.device, dt: float) -> None:
        """Set execution parameters and initialize gripper-specific tensors."""
        super().set_execution_params(num_envs, device, dt)

        self._action_dim_mask = self._create_action_mask("gripper")
        self._action_tensor = torch.zeros((self.num_envs, self.action_space_info.total_dim), device=self.device)
        self._interp_start_values = torch.zeros(self.num_envs, device=self.device)
        self._interp_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def _is_opening(self) -> bool:
        """Whether this action targets the open gripper state."""
        return self.target_value > 0.0

    @property
    def _position_bounds(self) -> tuple[float, float]:
        """Return closed/open position bounds in sorted order."""
        return min(self.closed_position, self.open_position), max(self.closed_position, self.open_position)

    def _target_command_value(self) -> float:
        """Return the environment command value for the requested logical gripper target."""
        if self.command_mode == "position":
            return self.open_position if self._is_opening else self.closed_position
        return self.target_value

    def _fallback_start_command(self) -> float:
        """Return a conservative start command when gripper observations are unavailable."""
        if self.command_mode == "position":
            return self.closed_position if self._is_opening else self.open_position
        return -self.target_value

    def _clamp_command(self, command_values: torch.Tensor) -> torch.Tensor:
        """Clamp commands to the valid range for the configured gripper command mode."""
        if self.command_mode == "position":
            lower, upper = self._position_bounds
            return command_values.clamp(lower, upper)
        if self._is_opening:
            return command_values.clamp(min=self._BINARY_EPS)
        return command_values.clamp(max=-self._BINARY_EPS)

    def _estimate_current_command(self, scene_data: SceneData, env_ids: torch.Tensor) -> torch.Tensor:
        """Estimate current gripper command in the configured command domain."""
        asset_name = self.agent_assets[0]
        robot_data = scene_data.articulations.get(asset_name)
        if robot_data is None or robot_data.gripper_pos is None:
            return torch.full(
                (len(env_ids),), self._fallback_start_command(), dtype=torch.float32, device=self.device
            )

        gripper_pos = robot_data.gripper_pos.to(self.device)
        # Panda observations store the two fingers with opposite signs, e.g. [0.04, -0.04].
        current_opening = gripper_pos[env_ids].abs().mean(dim=-1)
        if self.command_mode == "position":
            lower, upper = self._position_bounds
            return current_opening.clamp(lower, upper)

        denom = max(abs(self.open_position - self.closed_position), 1e-6)
        normalized = 2.0 * ((current_opening - self.closed_position) / denom) - 1.0
        return normalized.clamp(-1.0, 1.0)

    def _compute_action_impl(self, scene_data: SceneData, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute gripper action for controlled asset.

        Args:
            scene_data: Complete scene state container.
            env_ids: Indices of active environments.

        Returns:
            (action_tensor, action_dim_mask):
                - action_tensor: Shape (num_envs, action_dim) - action values for all envs
                - action_dim_mask: Shape (action_dim,) - which dimensions this action controls
        """
        assert self.action_space_info is not None, "action_space_info must be provided"

        self._action_tensor[env_ids] = 0.0
        target_command = self._target_command_value()
        command_values = torch.full((len(env_ids),), target_command, dtype=torch.float32, device=self.device)

        if self.interpolation_duration > 0.0:
            assert self._interp_initialized is not None
            assert self._interp_start_values is not None

            needs_init = ~self._interp_initialized[env_ids]
            if needs_init.any():
                init_env_ids = env_ids[needs_init]
                self._interp_start_values[init_env_ids] = self._estimate_current_command(scene_data, init_env_ids)
                self._interp_initialized[init_env_ids] = True

            alpha = ((self.time_elapsed[env_ids] - self.dt) / self.interpolation_duration).clamp(0.0, 1.0)
            start_values = self._interp_start_values[env_ids]
            command_values = start_values + alpha * (target_command - start_values)

        command_values = self._clamp_command(command_values)
        for idx in self._gripper_indices:
            self._action_tensor[env_ids, idx] = command_values

        return self._action_tensor, self._action_dim_mask

    def _check_completion_impl(self, scene_data: SceneData, env_ids: torch.Tensor) -> None:
        """Check if gripper action duration elapsed.

        Args:
            scene_data: Complete scene state container (unused for gripper).
            env_ids: Indices of active environments.
        """
        self._env_success_mask[env_ids] = self.time_elapsed[env_ids] >= max(
            self.duration, self.interpolation_duration
        )
        self._env_failure_mask[env_ids] = False

    def _reset_impl(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset interpolation state for selected environments."""
        if self._interp_initialized is None or self._interp_start_values is None:
            return
        if env_ids is None:
            self._interp_initialized.zero_()
            self._interp_start_values.zero_()
        else:
            self._interp_initialized[env_ids] = False
            self._interp_start_values[env_ids] = 0.0

    @classmethod
    def from_cfg(cls, cfg: GripperActionCfg):
        """Create gripper action from configuration."""
        return cls(
            agent_assets=cfg.agent_assets,
            target_value=cfg.target_value,
            duration=cfg.duration,
            timeout=cfg.timeout,
            interpolation_duration=cfg.interpolation_duration,
            command_mode=cfg.command_mode,
            closed_position=cfg.closed_position,
            open_position=cfg.open_position,
            action_space_info=cfg.action_space_info,
        )


class OpenGripper(GripperAction):
    """Open the gripper while holding the current pose."""

    cfg_type: ClassVar[type] = OpenGripperCfg

    def __init__(
        self,
        agent_assets: str | list[str],
        duration: float,
        timeout: float,
        target_value: float = 1.0,
        interpolation_duration: float = 0.0,
        command_mode: str | None = None,
        closed_position: float | None = None,
        open_position: float | None = None,
        action_space_info: ActionSpaceInfo | None = None,
    ):
        super().__init__(
            agent_assets=agent_assets,
            target_value=target_value,
            duration=duration,
            timeout=timeout,
            interpolation_duration=interpolation_duration,
            command_mode=command_mode,
            closed_position=closed_position,
            open_position=open_position,
            action_space_info=action_space_info,
        )


class CloseGripper(GripperAction):
    """Close the gripper while holding the current pose."""

    cfg_type: ClassVar[type] = CloseGripperCfg

    def __init__(
        self,
        agent_assets: str | list[str],
        duration: float,
        timeout: float,
        target_value: float = -1.0,
        interpolation_duration: float = 0.0,
        command_mode: str | None = None,
        closed_position: float | None = None,
        open_position: float | None = None,
        action_space_info: ActionSpaceInfo | None = None,
    ):
        super().__init__(
            agent_assets=agent_assets,
            target_value=target_value,
            duration=duration,
            timeout=timeout,
            interpolation_duration=interpolation_duration,
            command_mode=command_mode,
            closed_position=closed_position,
            open_position=open_position,
            action_space_info=action_space_info,
        )
