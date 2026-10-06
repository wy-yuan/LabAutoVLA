# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MoveToFrame action - move end-effector to an object's named frame."""

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import ClassVar

from .._compat import configclass
from ..math_utils import quat_mul, quat_rotate
from ..robot_action_spaces import ActionSpaceInfo
from ..scene_data import SceneData
from .move_to_pose import MoveToPose, MoveToPoseCfg


@configclass
class MoveToFrameCfg(MoveToPoseCfg):
    """Configuration for MoveToFrame action (move to object frame).

    Inherits from MoveToPoseCfg and adds object/frame specific fields.

    Attributes:
        object: Name of the object with the target frame. REQUIRED.
        frame: Name of the frame to move to (e.g., "grasp", "pre_grasp"). REQUIRED.
    """

    object: str = MISSING
    frame: str = MISSING
    position_noise_range: dict[str, tuple[float, float]] | None = None


class MoveToFrame(MoveToPose):
    """Move the end-effector to an object's named frame (offset) in world coordinates.

    Inherits from MoveToPose. On first call, looks up the frame pose from the object
    and uses it as the target. Subsequent calls use the cached target.
    """

    cfg_type: ClassVar[type] = MoveToFrameCfg

    def __init__(
        self,
        object: str,
        frame: str,
        agent_assets: str | list[str],
        timeout: float,
        position_threshold: float,
        orientation_threshold: float,
        interpolation_duration: float = 0.0,
        target_velocity: float | None = None,
        min_interp_duration: float = 0.1,
        max_interp_duration: float = 5.0,
        position_noise_range: dict[str, tuple[float, float]] | None = None,
        action_space_info: ActionSpaceInfo | None = None,
    ):
        """
        Args:
            object: Object name in scene_data.rigid_objects.
            frame: Named frame key under the object.
            agent_assets: Name(s) of articulated asset(s) acting as agents.
            timeout: Max time (in seconds) before timeout.
            position_threshold: Distance threshold for success (meters).
            orientation_threshold: Orientation threshold for success (radians).
            interpolation_duration: Time in seconds to ramp the commanded pose from current to target.
                                   Ignored when target_velocity is set.
            target_velocity: Optional constant end-effector speed (m/s) used to derive
                            interpolation_duration from the runtime start-to-target distance
                            (which includes any position_noise_range offset).
            min_interp_duration: Lower bound (seconds) for the derived duration.
            max_interp_duration: Upper bound (seconds) for the derived duration.
            position_noise_range: Optional per-axis uniform noise added once to the target frame.
            action_space_info: Optional action space metadata for mask creation.
        """
        # Initialize parent with None targets (will be set on first call)
        super().__init__(
            agent_assets=agent_assets,
            target_positions_w=None,
            target_orientations_w=None,
            timeout=timeout,
            position_threshold=position_threshold,
            orientation_threshold=orientation_threshold,
            interpolation_duration=interpolation_duration,
            target_velocity=target_velocity,
            min_interp_duration=min_interp_duration,
            max_interp_duration=max_interp_duration,
            action_space_info=action_space_info,
        )

        # Store frame lookup info
        self.object = object
        self.frame = frame
        self.position_noise_range = position_noise_range

    def _compute_action_impl(self, scene_data: SceneData, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute move-to-frame action for controlled asset.

        For envs newly entering this action, looks up the frame pose from the object and
        caches it as their target. Already-initialized envs (per _targets_initialized_mask,
        shared with the base class) keep their cached target.

        Environments advance through the action sequence independently (see
        StateMachine.step()), so this resolves the target per env_ids rather than using a
        single shared flag -- otherwise whichever env reaches this action first would fix
        the frame lookup (and noise draw) for every env, including ones that haven't
        actually arrived at this action yet.

        Args:
            scene_data: Complete scene state container.
            env_ids: Indices of active environments.

        Returns:
            (action_tensor, action_dim_mask):
                - action_tensor: Shape (num_envs, action_dim) - action values for all envs
                - action_dim_mask: Shape (action_dim,) - which dimensions this action controls
        """
        newly_initialized = env_ids[~self._targets_initialized_mask[env_ids]]
        if newly_initialized.numel() > 0:
            # Get target object data
            if self.object not in scene_data.rigid_objects:
                raise ValueError(
                    f"Object '{self.object}' not found in scene_data.rigid_objects. "
                    f"Available: {list(scene_data.rigid_objects.keys())}"
                )

            obj_data = scene_data.rigid_objects[self.object]

            # Get target frame pose (in object body frame)
            if obj_data.frames is None or self.frame not in obj_data.frames:
                available = list(obj_data.frames.keys()) if obj_data.frames else []
                raise ValueError(
                    f"Frame '{self.frame}' not found for object '{self.object}'. Available frames: {available}"
                )

            frame_pose = obj_data.frames[self.frame]
            n = newly_initialized.numel()

            # Frames are already in world frame (transformed by Isaac Lab's FrameTransformer)
            grasp_pos_w = frame_pose.position.to(self.device)[newly_initialized]
            grasp_quat_w = (
                frame_pose.orientation.to(self.device)[newly_initialized]
                if frame_pose.orientation is not None
                else None
            )
            if self.position_noise_range is not None:
                ranges = torch.tensor(
                    [self.position_noise_range.get(axis, (0.0, 0.0)) for axis in ("x", "y", "z")],
                    dtype=torch.float32,
                    device=self.device,
                )
                noise = ranges[:, 0] + torch.rand((n, 3), device=self.device) * (ranges[:, 1] - ranges[:, 0])
                grasp_pos_w = grasp_pos_w + noise

            # Apply robot-specific grasp-to-EE offset: ^W T_ee = ^W T_g · ^g T_ee
            if self.action_space_info and self.action_space_info.grasp_to_ee_offset and grasp_quat_w is not None:
                offset_pos, offset_quat = self.action_space_info.grasp_to_ee_offset
                offset_pos_t = (
                    torch.tensor(offset_pos, device=self.device, dtype=torch.float32).unsqueeze(0).expand(n, -1)
                )
                offset_quat_t = (
                    torch.tensor(offset_quat, device=self.device, dtype=torch.float32).unsqueeze(0).expand(n, -1)
                )

                self.target_positions_w[newly_initialized] = grasp_pos_w + quat_rotate(grasp_quat_w, offset_pos_t)
                self.target_orientations_w[newly_initialized] = quat_mul(grasp_quat_w, offset_quat_t)
            else:
                self.target_positions_w[newly_initialized] = grasp_pos_w
                if grasp_quat_w is not None:
                    self.target_orientations_w[newly_initialized] = grasp_quat_w

            # Shared with the base class: marks these envs as fully targeted so
            # MoveToPose._compute_action_impl's own (generic) fallback-fill skips them.
            self._targets_initialized_mask[newly_initialized] = True

        # Delegate to parent's implementation
        return super()._compute_action_impl(scene_data, env_ids)

    @classmethod
    def from_cfg(cls, cfg: MoveToFrameCfg):
        """Create MoveToFrame action from configuration."""
        return cls(
            object=cfg.object,
            frame=cfg.frame,
            agent_assets=cfg.agent_assets,
            timeout=cfg.timeout,
            position_threshold=cfg.position_threshold,
            orientation_threshold=cfg.orientation_threshold,
            interpolation_duration=cfg.interpolation_duration,
            target_velocity=cfg.target_velocity,
            min_interp_duration=cfg.min_interp_duration,
            max_interp_duration=cfg.max_interp_duration,
            position_noise_range=cfg.position_noise_range,
            action_space_info=cfg.action_space_info,
        )
