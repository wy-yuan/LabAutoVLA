# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MoveToPose action - move end-effector to target world pose (position + orientation)."""

from __future__ import annotations

import torch
from typing import ClassVar

from .._compat import configclass
from ..primitive_action import PrimitiveAction, PrimitiveActionCfg
from ..robot_action_spaces import ActionSpaceInfo
from ..scene_data import SceneData


def _quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize quaternions in (w, x, y, z) format."""
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def _quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Spherical interpolation between quaternions in (w, x, y, z) format."""
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)

    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.abs(dot).clamp(0.0, 1.0)

    t = t.unsqueeze(-1)
    linear = _quat_normalize(q0 + t * (q1 - q0))

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0).clamp_min(1e-6)
    theta = theta_0 * t
    s0 = torch.sin(theta_0 - theta) / sin_theta_0
    s1 = torch.sin(theta) / sin_theta_0
    spherical = _quat_normalize(s0 * q0 + s1 * q1)

    return torch.where(dot > 0.9995, linear, spherical)


@configclass
class MoveToPoseCfg(PrimitiveActionCfg):
    """Configuration for MoveToPose action (move to target pose).

    Inherits base fields (assets, timeout) from PrimitiveActionCfg.

    Attributes:
        target_positions: Target positions in world frame, shape (num_envs, 3).
                         If None, uses current robot position (position-only hold).
        target_orientations: Target orientation quaternions in world frame, shape (num_envs, 4) as (w,x,y,z).
                            If None, uses current robot orientation (orientation-only hold).
        position_threshold: Distance threshold for success (meters).
        orientation_threshold: Orientation threshold for success (radians).
        settling_time: Time (in seconds) the robot must remain within threshold before success.
                      Prevents false positives from overshoots or oscillations.
        interpolation_duration: Time (in seconds) used to ramp from the current EE pose to
                      the target pose. A value of 0.0 preserves the old immediate-target behavior.
                      Ignored when target_velocity is set.
        target_velocity: Optional constant end-effector speed (m/s). When set, the actual
                      interpolation_duration is derived at runtime from the start-to-target
                      distance (duration = distance / target_velocity) instead of using the
                      fixed interpolation_duration above, so commanded speed stays constant
                      even when the target position is randomized. Clamped to
                      [min_interp_duration, max_interp_duration].
        min_interp_duration: Lower bound (seconds) applied when deriving duration from
                      target_velocity.
        max_interp_duration: Upper bound (seconds) applied when deriving duration from
                      target_velocity.
    """

    target_positions: torch.Tensor | None = None
    target_orientations: torch.Tensor | None = None
    position_threshold: float = 0.01  # 10mm tolerance
    orientation_threshold: float = 0.02  # ~1.15 degrees tolerance (realistic for IK)
    settling_time: float = 0.05  # 50ms default (tunable per task)
    interpolation_duration: float = 0.0
    target_velocity: float | None = None
    min_interp_duration: float = 0.1
    max_interp_duration: float = 5.0


class MoveToPose(PrimitiveAction):
    """Move the end-effector to target world pose (position + orientation).

    Controls position and orientation only, does not modify gripper state.

    Supports partial specification:
    - target_positions_w=None: Hold current position (orientation-only control)
    - target_orientations_w=None: Hold current orientation (position-only control)
    - Both specified: Full pose control (default behavior)
    """

    cfg_type: ClassVar[type] = MoveToPoseCfg

    def __init__(
        self,
        agent_assets: str | list[str],
        target_positions_w: torch.Tensor | None,
        target_orientations_w: torch.Tensor | None,
        timeout: float,
        position_threshold: float,
        orientation_threshold: float,
        settling_time: float = 0.05,
        interpolation_duration: float = 0.0,
        target_velocity: float | None = None,
        min_interp_duration: float = 0.1,
        max_interp_duration: float = 5.0,
        action_space_info: ActionSpaceInfo | None = None,
    ):
        """
        Args:
            agent_assets: Name(s) of articulated asset(s) acting as agents.
            target_positions_w: (num_envs, 3) world-frame target positions.
                               If None, uses current robot position on first call.
            target_orientations_w: (num_envs, 4) world-frame target orientations as quaternions (w,x,y,z).
                                  If None, uses current robot orientation on first call.
            timeout: Max time (in seconds) before timeout.
            position_threshold: Distance threshold for success (meters).
            orientation_threshold: Orientation threshold for success (radians).
            settling_time: Time (in seconds) the robot must remain within threshold before success.
            interpolation_duration: Time in seconds to ramp the commanded pose from current to target.
                                   Ignored when target_velocity is set.
            target_velocity: Optional constant end-effector speed (m/s) used to derive
                            interpolation_duration from the runtime start-to-target distance.
            min_interp_duration: Lower bound (seconds) for the derived duration.
            max_interp_duration: Upper bound (seconds) for the derived duration.
            action_space_info: Optional action space metadata for mask creation.
        """
        super().__init__(agent_assets, timeout, action_space_info)

        # Track whether targets were originally None (for proper reset behavior)
        self._position_was_none = target_positions_w is None
        self._orientation_was_none = target_orientations_w is None

        # Store target tensors (will be moved to device in set_execution_params)
        self._target_positions_w_init = target_positions_w
        self._target_orientations_w_init = target_orientations_w

        # These will be set in set_execution_params()
        self.target_positions_w = None
        self.target_orientations_w = None

        self.position_threshold = position_threshold
        self.orientation_threshold = orientation_threshold
        self.settling_time = settling_time
        # Base (per-config, not per-env) duration; converted to a per-env tensor in
        # set_execution_params() so target_velocity can derive a distinct duration per env.
        self.interpolation_duration = interpolation_duration
        self.target_velocity = target_velocity
        self.min_interp_duration = min_interp_duration
        self.max_interp_duration = max_interp_duration
        # Whether this action ramps at all -- a static, per-config decision (not per-env),
        # so it's safe to use directly in an `if` even once interpolation_duration becomes
        # a per-env tensor below.
        self._use_interpolation = (target_velocity is not None) or (interpolation_duration > 0.0)

        # Settling time tracking (initialized in set_execution_params)
        self.time_in_threshold = None
        self._interp_start_positions_w = None
        self._interp_start_orientations_w = None

        # Validate action_space_info at init (fail-fast)
        if self.action_space_info is None:
            raise ValueError(
                "MoveToPose requires action_space_info to determine position/orientation indices. "
                "Pass action_space_info parameter when creating the action."
            )

        # Cache indices for fast access
        self._position_indices = (
            list(self.action_space_info.position_indices) if self.action_space_info.position_indices is not None else []
        )
        self._orientation_indices = (
            list(self.action_space_info.orientation_indices)
            if self.action_space_info.orientation_indices is not None
            else []
        )

        # These will be initialized in set_execution_params()
        self._action_dim_mask = None
        self._action_tensor = None

    def set_execution_params(self, num_envs: int, device: str | torch.device, dt: float) -> None:
        """Set execution parameters and initialize move-specific tensors."""
        super().set_execution_params(num_envs, device, dt)

        # Validate and move target tensors to device. When a target wasn't provided at
        # construction time, allocate a real (num_envs, ...) placeholder -- rather than
        # leaving it as a bare None -- so it can be filled in per-env (see _targets_initialized_mask
        # below) instead of racing across environments that reach this action at different times.
        if self._target_positions_w_init is not None:
            assert self._target_positions_w_init.shape == (
                num_envs,
                3,
            ), f"target_positions_w must have shape (num_envs, 3), got {self._target_positions_w_init.shape}"
            self.target_positions_w = self._target_positions_w_init.to(self.device)
        else:
            self.target_positions_w = torch.zeros((num_envs, 3), device=self.device)

        if self._target_orientations_w_init is not None:
            assert self._target_orientations_w_init.shape == (
                num_envs,
                4,
            ), f"target_orientations_w must have shape (num_envs, 4), got {self._target_orientations_w_init.shape}"
            self.target_orientations_w = self._target_orientations_w_init.to(self.device)
        else:
            self.target_orientations_w = torch.zeros((num_envs, 4), device=self.device)
            self.target_orientations_w[:, 0] = 1.0  # identity quaternion placeholder

        # Per-env flag: has this env's target_positions_w/target_orientations_w been resolved
        # for the current activation of this action? Environments run this state machine
        # asynchronously (each env has its own current_action_idx in StateMachine), so this
        # MUST be tracked per env rather than as a single shared flag -- otherwise whichever
        # env reaches this action first would latch a target/start pose for every env, using
        # the OTHER (not-yet-arrived) envs' stale current pose from an earlier action.
        # Subclasses (MoveToFrame, MoveRelative) that resolve target_positions_w themselves
        # (frame lookup, offset) share this same mask: they mark env ids as initialized once
        # they've written the target, which causes the generic fallback below to skip them.
        self._targets_initialized_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        if self._target_positions_w_init is not None and self._target_orientations_w_init is not None:
            # Both targets were fully specified at construction time (same for every env);
            # nothing left to lazily resolve.
            self._targets_initialized_mask[:] = True

        # Interpolation start pose and per-env derived duration, same per-env-latch reasoning
        # as _targets_initialized_mask above.
        self._interp_start_positions_w = torch.zeros((num_envs, 3), device=self.device)
        self._interp_start_orientations_w = torch.zeros((num_envs, 4), device=self.device)
        self._interp_initialized_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.interpolation_duration = torch.full(
            (num_envs,), float(self.interpolation_duration), dtype=torch.float32, device=self.device
        )

        # Cache the action dimension mask (computed once, reused every step)
        self._action_dim_mask = self._create_action_mask("position_orientation")

        # Pre-allocate action tensor (will be zeroed and reused each step)
        self._action_tensor = torch.zeros((self.num_envs, self.action_space_info.total_dim), device=self.device)

        # Initialize settling time tracker
        self.time_in_threshold = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Cache asset name (single-agent for now)
        self._asset_name = self.agent_assets[0]

    def _compute_action_impl(self, scene_data: SceneData, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute move to pose action for controlled asset.

        Args:
            scene_data: Complete scene state container.
            env_ids: Indices of active environments currently executing this action.

        Returns:
            (action_tensor, action_dim_mask):
                - action_tensor: Shape (num_envs, action_dim) - action values for all envs
                - action_dim_mask: Shape (action_dim,) - which dimensions this action controls
        """
        import time

        compute_start = time.perf_counter()
        timings = {}

        # Get robot articulation data (use cached asset name)
        data_access_start = time.perf_counter()
        if self._asset_name not in scene_data.articulations:
            raise ValueError(
                f"Asset '{self._asset_name}' not found in scene_data.articulations. "
                f"Available: {list(scene_data.articulations.keys())}"
            )

        robot_data = scene_data.articulations[self._asset_name]
        timings["data_access"] = time.perf_counter() - data_access_start

        # For envs newly entering this action (i.e. not yet resolved for this activation),
        # fill in any target left unspecified at construction time, using THEIR OWN current
        # pose. Only touches env_ids currently active on this action -- see
        # _targets_initialized_mask setup in set_execution_params for why this must be
        # per-env rather than a single shared flag.
        init_start = time.perf_counter()
        newly_targeted = env_ids[~self._targets_initialized_mask[env_ids]]
        if newly_targeted.numel() > 0:
            if self._position_was_none:
                if robot_data.ee_pos_w is None:
                    raise ValueError(f"End-effector position not available for asset '{self._asset_name}'")
                self.target_positions_w[newly_targeted] = robot_data.ee_pos_w[newly_targeted].to(self.device)

            if self._orientation_was_none:
                if robot_data.ee_quat_w is None:
                    raise ValueError(f"End-effector orientation not available for asset '{self._asset_name}'")
                self.target_orientations_w[newly_targeted] = robot_data.ee_quat_w[newly_targeted].to(self.device)

            self._targets_initialized_mask[newly_targeted] = True
        timings["target_init"] = time.perf_counter() - init_start

        # Optional command interpolation. The success check still uses the final
        # target, while the IK command ramps smoothly toward that target.
        command_positions_w = self.target_positions_w
        command_orientations_w = self.target_orientations_w
        if self._use_interpolation:
            # Capture each env's interpolation start pose exactly when IT first becomes
            # active on this action (env_ids), not whenever the first env in the whole
            # batch happens to reach it -- environments advance through the action
            # sequence asynchronously (see StateMachine.step()), so a shared/global latch
            # would capture late-arriving envs' stale mid-previous-action pose instead.
            newly_started = env_ids[~self._interp_initialized_mask[env_ids]]
            if newly_started.numel() > 0:
                self._interp_start_positions_w[newly_started] = robot_data.ee_pos_w[newly_started].to(self.device)
                self._interp_start_orientations_w[newly_started] = robot_data.ee_quat_w[newly_started].to(
                    self.device
                )

                if self.target_velocity is not None:
                    # Derive duration from the actual start-to-target distance so the
                    # commanded speed (and resulting tracking lag) stays constant even
                    # when the target position is randomized (e.g. position_noise_range).
                    distance = (
                        self.target_positions_w[newly_started] - self._interp_start_positions_w[newly_started]
                    ).norm(dim=-1)
                    self.interpolation_duration[newly_started] = (distance / self.target_velocity).clamp(
                        self.min_interp_duration, self.max_interp_duration
                    )

                self._interp_initialized_mask[newly_started] = True

            alpha = ((self.time_elapsed - self.dt) / self.interpolation_duration).clamp(0.0, 1.0)
            alpha_pos = alpha.unsqueeze(-1)
            command_positions_w = (
                self._interp_start_positions_w
                + alpha_pos * (self.target_positions_w - self._interp_start_positions_w)
            )
            command_orientations_w = _quat_slerp(
                self._interp_start_orientations_w,
                self.target_orientations_w,
                alpha,
            )

        # Zero the pre-allocated action tensor (reuse memory)
        zero_start = time.perf_counter()
        self._action_tensor.zero_()
        timings["tensor_zero"] = time.perf_counter() - zero_start

        # Convert target pose (position + orientation) from world frame to robot base frame
        # Target is specified in world frame, action is in robot base frame
        frame_convert_start = time.perf_counter()
        target_positions_b, target_orientations_b = self._convert_world_to_base_frame(
            scene_data,
            self._asset_name,
            command_positions_w,
            command_orientations_w,
        )
        timings["frame_conversion"] = time.perf_counter() - frame_convert_start

        # Set position dimensions using cached indices (only for active environments)
        tensor_fill_start = time.perf_counter()
        for i, idx in enumerate(self._position_indices):
            self._action_tensor[env_ids, idx] = target_positions_b[env_ids, i]

        # Set orientation dimensions using cached indices (only for active environments)
        for i, idx in enumerate(self._orientation_indices):
            self._action_tensor[env_ids, idx] = target_orientations_b[env_ids, i]
        timings["tensor_fill"] = time.perf_counter() - tensor_fill_start

        # Note: Gripper state is NOT modified by this action (controlled by mask)

        total_time = time.perf_counter() - compute_start

        # Print timing breakdown if significantly slow (>5ms)
        if total_time > 0.005:
            RED = "\033[91m"
            YELLOW = "\033[93m"
            RESET = "\033[0m"
            print(f"\n{RED}{'─' * 80}")
            print(f"⚠️  SLOW ACTION COMPUTE: {self.__class__.__name__} took {total_time * 1000:.2f}ms")
            print(f"{'─' * 80}{RESET}")
            print(f"{YELLOW}  Action timing breakdown:{RESET}")
            sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)
            for name, timing_val in sorted_timings:
                timing_ms = timing_val * 1000
                percent = (timing_val / total_time) * 100 if total_time > 0 else 0
                print(f"{YELLOW}    ▸ {name}: {timing_ms:.2f}ms ({percent:.1f}%){RESET}")
            print(f"{RED}{'─' * 80}{RESET}\n")

        return self._action_tensor, self._action_dim_mask

    def _check_completion_impl(self, scene_data: SceneData, env_ids: torch.Tensor) -> None:
        """Check if target pose reached (both position and orientation within thresholds).

        Args:
            scene_data: Complete scene state container.
            env_ids: Indices of active environments.
        """
        # Skip checking envs whose target hasn't been resolved yet this activation (their
        # first step on this action: _compute_action_impl resolves the target AFTER this
        # method runs, see PrimitiveAction.compute_action). Per-env, since other envs may
        # already be mid-motion on this same action instance.
        uninitialized_env_ids = env_ids[~self._targets_initialized_mask[env_ids]]
        if uninitialized_env_ids.numel() > 0:
            self._env_success_mask[uninitialized_env_ids] = False
            self._env_failure_mask[uninitialized_env_ids] = False

        env_ids = env_ids[self._targets_initialized_mask[env_ids]]
        if env_ids.numel() == 0:
            return

        # Get robot articulation data
        if self._asset_name not in scene_data.articulations:
            raise ValueError(
                f"Asset '{self._asset_name}' not found in scene_data.articulations. "
                f"Available: {list(scene_data.articulations.keys())}"
            )

        robot_data = scene_data.articulations[self._asset_name]

        # Validate end-effector data is available
        if robot_data.ee_pos_w is None:
            raise ValueError(f"End-effector position not available for asset '{self._asset_name}'")
        if robot_data.ee_quat_w is None:
            raise ValueError(f"End-effector orientation not available for asset '{self._asset_name}'")

        # Position distance check
        current_pos = robot_data.ee_pos_w.to(self.device)
        position_distance = torch.norm(self.target_positions_w - current_pos, dim=1)
        position_reached = position_distance < self.position_threshold

        # Orientation distance check using quaternion dot product
        # Formula: angular_distance = 2 * arccos(|dot(q1, q2)|)
        # This gives the geodesic distance on SO(3) in radians
        # The abs() handles quaternion double cover (q and -q represent same rotation)
        current_quat = robot_data.ee_quat_w.to(self.device)
        dot_product = torch.sum(self.target_orientations_w * current_quat, dim=1)
        # Clamp to [0, 1] to avoid numerical issues with arccos
        dot_product_clamped = torch.clamp(torch.abs(dot_product), 0.0, 1.0)
        angular_distance = 2.0 * torch.acos(dot_product_clamped)
        orientation_reached = angular_distance < self.orientation_threshold

        # Check if both position and orientation are within thresholds
        within_threshold = position_reached & orientation_reached

        # Update settling timer for active environments
        # If within threshold: accumulate time, else reset to zero
        self.time_in_threshold[env_ids] = torch.where(
            within_threshold[env_ids],
            self.time_in_threshold[env_ids] + self.dt,  # Accumulate
            torch.zeros_like(self.time_in_threshold[env_ids]),  # Reset
        )

        # Success only if settled for required duration
        self._env_success_mask[env_ids] = self.time_in_threshold[env_ids] >= self.settling_time

        # No custom failure modes
        self._env_failure_mask[env_ids] = False

    def _reset_impl(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset per-env initialization flags and settling timers for the given environments.

        Only touches the given env_ids (or all envs, if None) -- this must stay per-env
        so that resetting some environments (e.g. one env finishing/restarting an episode)
        never clears already-valid state for other environments still mid-action.

        Args:
            env_ids: Indices of environments being reset, or None for all.
        """
        if env_ids is None:
            self.time_in_threshold.zero_()
            self._targets_initialized_mask.zero_()
            self._interp_initialized_mask.zero_()
        else:
            self.time_in_threshold[env_ids] = 0.0
            self._targets_initialized_mask[env_ids] = False
            self._interp_initialized_mask[env_ids] = False

    @classmethod
    def from_cfg(cls, cfg: MoveToPoseCfg):
        """Create MoveToPose action from configuration."""
        return cls(
            agent_assets=cfg.agent_assets,
            target_positions_w=cfg.target_positions,
            target_orientations_w=cfg.target_orientations,
            timeout=cfg.timeout,
            position_threshold=cfg.position_threshold,
            orientation_threshold=cfg.orientation_threshold,
            settling_time=cfg.settling_time,
            interpolation_duration=cfg.interpolation_duration,
            target_velocity=cfg.target_velocity,
            min_interp_duration=cfg.min_interp_duration,
            max_interp_duration=cfg.max_interp_duration,
            action_space_info=cfg.action_space_info,
        )
