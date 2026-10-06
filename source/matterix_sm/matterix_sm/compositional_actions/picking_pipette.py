# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PickingPipette compositional action for lifting a pipette from its rack."""

from __future__ import annotations

from dataclasses import MISSING, field

from .._compat import configclass
from ..compositional_action import CompositionalActionCfg
from ..primitive_actions import CloseGripperCfg, MoveToFrameCfg, OpenGripperCfg
from ..robot_action_spaces import ActionSpaceInfo


def _pipette_move_to_frame(
    pipette: str,
    frame: str,
    agent_assets: str | list[str],
    action_space_info: ActionSpaceInfo | None,
    interpolation_duration: float,
    position_noise_range: dict[str, tuple[float, float]] | None = None,
    position_threshold: float = 0.02,
    target_velocity: float | None = None,
    settling_time: float = 0.05,
) -> MoveToFrameCfg:
    """Create a frame move with the tolerances used for pipette manipulation."""
    return MoveToFrameCfg(
        object=pipette,
        frame=frame,
        agent_assets=agent_assets,
        action_space_info=action_space_info,
        position_threshold=position_threshold,
        orientation_threshold=0.1,
        interpolation_duration=interpolation_duration,
        target_velocity=target_velocity,
        position_noise_range=position_noise_range,
        settling_time=settling_time,
    )


@configclass
class PickingPipetteCfg(CompositionalActionCfg):
    """Move to a pipette, grasp it, and lift it from the rack."""

    agent_assets: str | list[str] = MISSING
    pipette: str = MISSING
    action_space_info: ActionSpaceInfo | None = None

    interpolation_duration: float = 0.8
    gripper_interpolation_duration: float = 1.2
    grasp_position_threshold: float = 0.005
    # Constant end-effector speeds (m/s), split by how precision-sensitive the move is. Both
    # override interpolation_duration for their moves: duration is derived at runtime from the
    # actual start-to-target distance, keeping speed (and tracking lag) constant regardless of
    # position_noise_range. Either can be set to None to fall back to the fixed
    # interpolation_duration values instead.
    #
    # grasp_target_velocity: used ONLY for the final descent onto the pipette ("grasp" move),
    #   where tracking lag directly affects grasp success. Kept slow -- ~10mm/s keeps steady-state
    #   tracking lag under ~2mm given this robot's HIGH_PD gains (see grasp_position_threshold).
    # vertical_target_velocity: used for pure up/down moves that aren't the grasp itself --
    #   post_grasp (lifting the just-grasped pipette) here, plus the aspirate/dispense
    #   depth and lift moves in PipetteLiquidCfg. Slower than transit on purpose: lifting a
    #   freshly-grasped object too fast risks jerking it loose, and dipping into liquid too
    #   fast risks splashing/disturbing it -- neither is a concern for horizontal transit.
    # transit_target_velocity: used for horizontal/repositioning moves (pre_grasp here, plus
    #   the liquid_approach moves in PipetteLiquidCfg) -- these have generous position
    #   tolerance and no delicate contact, so they can run much faster without hurting grasp
    #   quality. Faster transit also keeps single-phase segment length from ballooning into
    #   long, low-information stretches that hurt BC training (compounding error, redundant frames).
    grasp_target_velocity: float | None = 0.1
    vertical_target_velocity: float | None = 0.12
    transit_target_velocity: float | None = 0.18
    # Minimum continuous time (seconds) the end-effector must stay within position_threshold
    # AND orientation_threshold before a move counts as successful (see MoveToPoseCfg). The
    # default of 0.05 is shorter than one control step at this robot's control rate (~0.083s,
    # 60Hz physics / decimation=5), so it was effectively a no-op -- moves succeeded on the very
    # first frame that touched the threshold, meaning the demonstration data never contained any
    # "settle and correct residual error" frames. Set explicitly here (0.25s, ~3 control steps)
    # so generated data actually includes that stabilization behavior for the policy to learn.
    settling_time: float = 0.25
    pre_grasp_position_noise_range: dict[str, tuple[float, float]] | None = field(
        default_factory=lambda: {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
        }
    )
    post_grasp_position_noise_range: dict[str, tuple[float, float]] | None = field(
        default_factory=lambda: {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
        }
    )

    def __post_init__(self):
        """Build the five primitive actions used by standalone and full workflows."""
        super().__post_init__()
        self.sub_actions = [
            _pipette_move_to_frame(
                pipette=self.pipette,
                frame="pre_grasp",
                agent_assets=self.agent_assets,
                action_space_info=self.action_space_info,
                interpolation_duration=self.interpolation_duration,
                position_noise_range=self.pre_grasp_position_noise_range,
                target_velocity=self.transit_target_velocity,
                settling_time=self.settling_time,
            ),
            OpenGripperCfg(
                agent_assets=self.agent_assets,
                duration=0.5,
                interpolation_duration=self.gripper_interpolation_duration,
                action_space_info=self.action_space_info,
            ),
            _pipette_move_to_frame(
                pipette=self.pipette,
                frame="grasp",
                agent_assets=self.agent_assets,
                action_space_info=self.action_space_info,
                interpolation_duration=self.interpolation_duration+0.4,
                position_threshold=self.grasp_position_threshold,
                target_velocity=self.grasp_target_velocity,
                settling_time=self.settling_time,
            ),
            CloseGripperCfg(
                agent_assets=self.agent_assets,
                duration=1.7,
                interpolation_duration=self.gripper_interpolation_duration, # 0.3s hold time
                action_space_info=self.action_space_info,
            ),
            _pipette_move_to_frame(
                pipette=self.pipette,
                frame="post_grasp",
                agent_assets=self.agent_assets,
                action_space_info=self.action_space_info,
                interpolation_duration=self.interpolation_duration,
                position_noise_range=self.post_grasp_position_noise_range,
                target_velocity=self.vertical_target_velocity,
                settling_time=self.settling_time,
            ),
        ]
