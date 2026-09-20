# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pipette liquid-transfer workflow built on the reusable pipette-picking action."""

from __future__ import annotations

from dataclasses import MISSING

from .._compat import configclass
from ..primitive_actions import MoveRelativeCfg, MoveToFrameCfg
from .picking_pipette import PickingPipetteCfg


@configclass
class PipetteLiquidCfg(PickingPipetteCfg):
    """Pick up a pipette, dip into the source, and dip into the target.

    The source and target assets must define a "liquid_approach" frame above
    their openings. Picking motion, gripper interpolation, and picking noise
    settings are inherited from PickingPipetteCfg.

    "aspirate_duration" and "dispense_duration" remain available for
    configuration compatibility; the current workflow has no hold primitives.
    """

    source: str = MISSING
    target: str = MISSING

    aspirate_depth: float = 0.25
    dispense_depth: float = 0.25
    aspirate_duration: float = 1.5
    dispense_duration: float = 1.5
    lift_height: float = 0.15

    def __post_init__(self):
        """Build the picking prefix, then append the liquid-transfer actions."""
        super().__post_init__()
        self.sub_actions.extend(
            [
                MoveToFrameCfg(
                    object=self.source,
                    frame="liquid_approach",
                    agent_assets=self.agent_assets,
                    interpolation_duration=self.interpolation_duration,
                    action_space_info=self.action_space_info,
                ),
                MoveRelativeCfg(
                    agent_assets=self.agent_assets,
                    position_offset=(0.0, 0.0, -self.aspirate_depth),
                    orientation_offset=None,
                    interpolation_duration=self.interpolation_duration,
                    action_space_info=self.action_space_info,
                ),
                MoveRelativeCfg(
                    agent_assets=self.agent_assets,
                    position_offset=(0.0, 0.0, self.lift_height),
                    orientation_offset=None,
                    interpolation_duration=self.interpolation_duration,
                    action_space_info=self.action_space_info,
                ),
                MoveToFrameCfg(
                    object=self.target,
                    frame="liquid_approach",
                    agent_assets=self.agent_assets,
                    interpolation_duration=self.interpolation_duration,
                    action_space_info=self.action_space_info,
                ),
                MoveRelativeCfg(
                    agent_assets=self.agent_assets,
                    position_offset=(0.0, 0.0, -self.dispense_depth),
                    orientation_offset=None,
                    interpolation_duration=self.interpolation_duration,
                    action_space_info=self.action_space_info,
                ),
                MoveRelativeCfg(
                    agent_assets=self.agent_assets,
                    position_offset=(0.0, 0.0, self.lift_height),
                    orientation_offset=None,
                    interpolation_duration=self.interpolation_duration,
                    action_space_info=self.action_space_info,
                ),
            ]
        )
