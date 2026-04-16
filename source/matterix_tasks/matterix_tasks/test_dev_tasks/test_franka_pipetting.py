# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test development environment for pipetting: pick up a pipette and transfer liquid
between a source beaker and a target beaker using a Franka robot."""

from matterix.envs import MatterixBaseEnvCfg, mdp
from matterix.managers import EventManagerCfg
from matterix.particle_systems import FluidCfg
from matterix_assets.infrastructure.tables import TABLE_SEATTLE_INST_Cfg
from matterix_assets.labware.beakers import BEAKER_500ML_INST_CFG
from matterix_assets.labware.pipettes import PIPETTE_1ML_INST_CFG, PIPETTE_RACK_CFG
from matterix_assets.robots import FRANKA_PANDA_HIGH_PD_IK_CFG

from matterix_sm import PipetteLiquidCfg
from matterix_sm.robot_action_spaces import FRANKA_IK_ACTION_SPACE

import isaaclab.envs.mdp as isaaclab_mdp
import isaaclab.sim as sim_utils
from isaaclab.sim import RenderCfg, SimulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass


##
# Asset variants with pipetting-specific frames
##

@configclass
class BEAKER_SOURCE_CFG(BEAKER_500ML_INST_CFG):
    """Source beaker: same geometry as the 500 mL beaker, with an added
    ``liquid_approach`` frame that tells the robot EE where to hover before
    dipping the pipette tip into the liquid.

    Frame height calibration (relative to beaker body origin):
      The 500 mL beaker is ~12 cm tall; its USD origin sits near the base.
      ``liquid_approach`` at z = 0.18 m places the robot EE ~6 cm above the
      beaker rim so the pipette tip (hanging ~15 cm below the EE) reaches the
      liquid surface inside. Adjust once the real USD dimensions are known.
    """

    frames = {
        # Inherited pick-up frames (unchanged)
        "pre_grasp": (0.0, 0.0, 0.04),
        "grasp": (0.0, 0.0, 0.0),
        "post_grasp": (0.0, 0.0, 0.05),
        # Pipetting frame: EE hover position for tip approach
        "liquid_approach": (0.0, 0.0, 0.18),
    }


@configclass
class BEAKER_TARGET_CFG(BEAKER_500ML_INST_CFG):
    """Target beaker: same as source beaker variant (same frame layout)."""

    frames = {
        "pre_grasp": (0.0, 0.0, 0.04),
        "grasp": (0.0, 0.0, 0.0),
        "post_grasp": (0.0, 0.0, 0.05),
        "liquid_approach": (0.0, 0.0, 0.18),
    }


##
# Event configs
##

@configclass
class EventCfg(EventManagerCfg):
    """Randomisation events for the pipetting environment."""

    reset_scene_to_default = EventTerm(
        func=isaaclab_mdp.reset_scene_to_default,
        mode="reset",
    )

    # Randomise beaker positions slightly so the workflow must generalise
    randomize_source_beaker = EventTerm(
        func=isaaclab_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("source_beaker"),
        },
    )

    randomize_target_beaker = EventTerm(
        func=isaaclab_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("target_beaker"),
        },
    )


##
# Observation configs
##

@configclass
class ObservationManagerCfg:
    """Observation specifications for the pipetting MDP."""

    @configclass
    class ArticulationsGroup(ObsGroup):
        """Robot state observations."""

        robot__root_world_pos = ObsTerm(func=mdp.root_world_pos, params={"asset_name": "robot"})
        robot__root_world_quat = ObsTerm(func=mdp.root_world_quat, params={"asset_name": "robot"})
        robot__joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_name": "robot"})
        robot__joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_name": "robot"})
        robot__ee_world_pos = ObsTerm(func=mdp.ee_world_pos, params={"asset_name": "robot"})
        robot__ee_world_quat = ObsTerm(func=mdp.ee_world_quat, params={"asset_name": "robot"})
        robot__gripper_pos = ObsTerm(func=mdp.gripper_pos, params={"asset_name": "robot"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RigidObjectsGroup(ObsGroup):
        """Pipette and beaker state observations."""

        # Pipette pose
        pipette__object_world_pos = ObsTerm(
            func=mdp.object_world_pos, params={"asset_name": "pipette"}
        )
        pipette__object_world_quat = ObsTerm(
            func=mdp.object_world_quat, params={"asset_name": "pipette"}
        )
        # All frames used by PickObjectCfg (pre_grasp → grasp → post_grasp)
        pipette__pre_grasp_frame = ObsTerm(
            func=mdp.frame_world_pose,
            params={"asset_name": "pipette", "frame_name": "pre_grasp"},
        )
        pipette__grasp_frame = ObsTerm(
            func=mdp.frame_world_pose,
            params={"asset_name": "pipette", "frame_name": "grasp"},
        )
        pipette__post_grasp_frame = ObsTerm(
            func=mdp.frame_world_pose,
            params={"asset_name": "pipette", "frame_name": "post_grasp"},
        )

        # Source beaker
        source_beaker__object_world_pos = ObsTerm(
            func=mdp.object_world_pos, params={"asset_name": "source_beaker"}
        )
        source_beaker__liquid_approach_frame = ObsTerm(
            func=mdp.frame_world_pose,
            params={"asset_name": "source_beaker", "frame_name": "liquid_approach"},
        )

        # Target beaker
        target_beaker__object_world_pos = ObsTerm(
            func=mdp.object_world_pos, params={"asset_name": "target_beaker"}
        )
        target_beaker__liquid_approach_frame = ObsTerm(
            func=mdp.frame_world_pose,
            params={"asset_name": "target_beaker", "frame_name": "liquid_approach"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class CameraGroup(ObsGroup):
        """Camera image observations for VLA training."""

        overhead_rgb = ObsTerm(
            func=mdp.camera_rgb,
            params={"sensor_name": "overhead_camera"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    articulations: ArticulationsGroup = ArticulationsGroup()
    rigid_objects: RigidObjectsGroup = RigidObjectsGroup()
    camera: CameraGroup = CameraGroup()


##
# Task environment config
##

@configclass
class FrankaPipettingEnvTestCfg(MatterixBaseEnvCfg):
    """Pipetting task: Franka picks up a pipette and transfers fluid from
    a source beaker to a target beaker.

    Scene layout (top-down, all positions in metres):
        Robot      at (0.0,  0.0, 0.0)
        Table      at (0.5,  0.0, 0.0)
        Pipette    at (0.55, 0.3, 0.05)  — on right side of table
        Source     at (0.6,  0.1, 0.05)  — centre-right of table
        Target     at (0.6, -0.1, 0.05)  — centre-left of table

    Particle system:
        A FluidCfg is pre-loaded inside the source beaker so that liquid
        dynamics are visible during the transfer.
    """
    # sim: SimulationCfg = SimulationCfg(
    #     # dt=1/60.0,
    #     render=RenderCfg(rendering_mode="performance"),
    # )
    
    env_spacing = 6.0

    # ── Static objects ───────────────────────────────────────────────────────
    objects = {
        # "pipette": PIPETTE_1ML_INST_CFG(pos=(0.6518, -0.275, 0.01)),
        # "pipette_rack": PIPETTE_RACK_CFG(pos=(0.65, -0.3, 0.0)),
        "pipette": PIPETTE_1ML_INST_CFG(pos=(0.55, -0.3, 0.05)),
        "pipette_rack": PIPETTE_RACK_CFG(pos=(0.55, -0.3, 0.0)),
        "source_beaker": BEAKER_SOURCE_CFG(pos=(0.6, 0.2, 0.05)),
        "target_beaker": BEAKER_TARGET_CFG(pos=(0.6, 0.0, 0.05)),
        "table": TABLE_SEATTLE_INST_Cfg(pos=(0.5, 0.0, 0.0)),
    }

    # ── Robots ───────────────────────────────────────────────────────────────
    articulated_assets = {
        "robot": FRANKA_PANDA_HIGH_PD_IK_CFG(pos=(0.0, 0.0, 0.0)),
    }

    # ── Particle systems ─────────────────────────────────────────────────────
    # Fluid pre-loaded inside the source beaker.
    # pos is in world frame; volume (x, y, z) fills the beaker interior.
    particle_systems = {
        "source_fluid": FluidCfg(
            pos=(0.6, 0.1, 0.12),          # slightly above beaker origin
            volume=(0.04, 0.04, 0.06),      # ~96 mL block of fluid
        ),
    }

    # ── Sensors (env-level, not attached to a specific asset) ─────────────
    sensors = {
        "overhead_camera": TiledCameraCfg(
            prim_path="/World/envs/env_.*/OverheadCamera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.9, 0.0, 0.4),
                rot=(0, -0.258819, 0, 0.9659258),
                convention="world",
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=12.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 20.0),
            ),
            width=224,
            height=224,
        ),
    }

    gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]

    # Re-render after resets so the camera sees the freshly-reset scene
    rerender_on_reset = True

    observations = ObservationManagerCfg()
    events = EventCfg()

    record_path = "datasets/pipetting_dataset.hdf5"

    # ── Workflows ────────────────────────────────────────────────────────────
    workflows = {
        "pipette_liquid": PipetteLiquidCfg(
            description=(
                "Pick up the pipette, aspirate from the source beaker, "
                "and dispense into the target beaker."
            ),
            agent_assets="robot",
            pipette="pipette",
            source="source_beaker",
            target="target_beaker",
            aspirate_depth=0.14,
            dispense_depth=0.14,
            aspirate_duration=1.5,
            dispense_duration=1.5,
            lift_height=0.18,
            action_space_info=FRANKA_IK_ACTION_SPACE,
        ),
    }
