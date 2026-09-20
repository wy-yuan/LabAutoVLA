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
        position_noise_range=position_noise_range,
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
            ),
        ]
