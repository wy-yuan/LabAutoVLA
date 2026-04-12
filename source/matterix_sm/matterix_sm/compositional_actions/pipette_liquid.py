# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PipetteLiquid compositional action - pick up a pipette and transfer liquid between containers."""

from __future__ import annotations

from dataclasses import MISSING

from .._compat import configclass
from ..compositional_action import CompositionalActionCfg
from ..primitive_actions import (
    OpenGripperCfg,
    CloseGripperCfg,
    MoveRelativeCfg,
    MoveToFrameCfg,
)
from ..robot_action_spaces import ActionSpaceInfo
from .pick_object import PickObjectCfg


def _relaxed_move_to_frame(object: str, frame: str, agent_assets, action_space_info) -> MoveToFrameCfg:
    """Create a MoveToFrameCfg with relaxed thresholds for pipetting tasks."""
    cfg = MoveToFrameCfg()
    cfg.object = object
    cfg.frame = frame
    cfg.agent_assets = agent_assets
    cfg.action_space_info = action_space_info
    cfg.position_threshold = 0.02      # 20mm (default: 10mm)
    cfg.orientation_threshold = 0.1    # ~5.7° (default: ~1.15°)
    return cfg


@configclass
class PipetteLiquidCfg(CompositionalActionCfg):
    """Configuration for PipetteLiquid compositional action.

    Transfers liquid from a source container to a target container using a pipette.

    Full sequence:
        PickObject(pipette)
        → MoveToFrame(source, "liquid_approach")   # position EE above source opening
        → MoveRelative(-aspirate_depth in z)        # lower tip into liquid
        → CloseGripper(aspirate_duration)           # hold — aspirate wait
        → MoveRelative(+lift_height in z)           # lift tip clear of source
        → MoveToFrame(target, "liquid_approach")   # position EE above target opening
        → MoveRelative(-dispense_depth in z)        # lower tip into target
        → CloseGripper(dispense_duration)           # hold — dispense wait
        → MoveRelative(+lift_height in z)           # lift tip clear of target

    The source and target beaker assets must define a "liquid_approach" frame that
    positions the robot EE above the container opening, at the correct height for
    the pipette tip to reach the liquid surface after the aspirate/dispense dip.

    Attributes:
        agent_assets: Name of the robot executing the workflow. REQUIRED.
        pipette: Name of the pipette object in the scene. REQUIRED.
        source: Name of the source (liquid donor) container. REQUIRED.
        target: Name of the target (liquid recipient) container. REQUIRED.
        aspirate_depth: Downward dip distance (m) into the source liquid. Default: 0.05.
        dispense_depth: Downward dip distance (m) into the target container. Default: 0.05.
        aspirate_duration: Time (s) to hold at the source dip position. Default: 1.5.
        dispense_duration: Time (s) to hold at the target dip position. Default: 1.5.
        lift_height: Upward clearance (m) after each dip. Default: 0.08.
        action_space_info: Action space metadata for the robot. REQUIRED.
    """

    # Required fields
    agent_assets: str | list[str] = MISSING
    pipette: str = MISSING
    source: str = MISSING
    target: str = MISSING
    action_space_info: ActionSpaceInfo | None = None

    # Tunable motion parameters
    aspirate_depth: float = 0.05      # m — how far to dip into source liquid
    dispense_depth: float = 0.05      # m — how far to dip into target
    aspirate_duration: float = 1.5    # s — hold time to simulate aspiration
    dispense_duration: float = 1.5    # s — hold time to simulate dispensing
    lift_height: float = 0.08         # m — clearance height after each dip

    def __post_init__(self):
        """Build the 9-step primitive action sequence after field initialisation."""
        super().__post_init__()

        self.sub_actions = [
            # ── Step 1: Pick up the pipette ──────────────────────────────────
            # PickObjectCfg(
            #     description="Pick up pipette from rack",
            #     agent_assets=self.agent_assets,
            #     object=self.pipette,
            #     action_space_info=self.action_space_info,
            # ),
             OpenGripperCfg(
                # target_value=0.1, # open gripper to 20% for pick-up
                agent_assets=self.agent_assets,
                action_space_info=self.action_space_info,
            ),
            _relaxed_move_to_frame(
                object=self.pipette,
                frame="pre_grasp",
                agent_assets=self.agent_assets,
                action_space_info=self.action_space_info,
            ),

            _relaxed_move_to_frame(
                object=self.pipette,
                frame="grasp",
                agent_assets=self.agent_assets,
                action_space_info=self.action_space_info,
            ),
            CloseGripperCfg(
                agent_assets=self.agent_assets,
                action_space_info=self.action_space_info,
            ),
            _relaxed_move_to_frame(
                object=self.pipette,
                frame="post_grasp",
                agent_assets=self.agent_assets,
                action_space_info=self.action_space_info,
            ),

            # ── Step 2: Move above source container ──────────────────────────
            # MoveToFrameCfg(
            #     object=self.source,
            #     frame="liquid_approach",
            #     agent_assets=self.agent_assets,
            #     action_space_info=self.action_space_info,
            # ),

            # # ── Step 3: Lower tip into source liquid ─────────────────────────
            # MoveRelativeCfg(
            #     agent_assets=self.agent_assets,
            #     position_offset=(0.0, 0.0, -self.aspirate_depth),
            #     orientation_offset=None,
            #     action_space_info=self.action_space_info,
            # ),

            # # ── Step 4: Hold position — simulate aspiration ──────────────────
            # # Keeps the gripper closed (holds the pipette) for aspirate_duration.
            # CloseGripperCfg(
            #     agent_assets=self.agent_assets,
            #     duration=self.aspirate_duration,
            #     action_space_info=self.action_space_info,
            # ),

            # # ── Step 5: Lift tip clear of source ─────────────────────────────
            # MoveRelativeCfg(
            #     agent_assets=self.agent_assets,
            #     position_offset=(0.0, 0.0, self.lift_height),
            #     orientation_offset=None,
            #     action_space_info=self.action_space_info,
            # ),

            # # ── Step 6: Move above target container ──────────────────────────
            # MoveToFrameCfg(
            #     object=self.target,
            #     frame="liquid_approach",
            #     agent_assets=self.agent_assets,
            #     action_space_info=self.action_space_info,
            # ),

            # # ── Step 7: Lower tip into target ────────────────────────────────
            # MoveRelativeCfg(
            #     agent_assets=self.agent_assets,
            #     position_offset=(0.0, 0.0, -self.dispense_depth),
            #     orientation_offset=None,
            #     action_space_info=self.action_space_info,
            # ),

            # # ── Step 8: Hold position — simulate dispensing ──────────────────
            # CloseGripperCfg(
            #     agent_assets=self.agent_assets,
            #     duration=self.dispense_duration,
            #     action_space_info=self.action_space_info,
            # ),

            # # ── Step 9: Lift tip clear of target ─────────────────────────────
            # MoveRelativeCfg(
            #     agent_assets=self.agent_assets,
            #     position_offset=(0.0, 0.0, self.lift_height),
            #     orientation_offset=None,
            #     action_space_info=self.action_space_info,
            # ),
        ]
