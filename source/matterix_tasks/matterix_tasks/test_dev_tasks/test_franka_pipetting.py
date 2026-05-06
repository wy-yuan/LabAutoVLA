# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test development environment for pipetting: pick up a pipette and transfer liquid
between a source beaker and a target beaker using a Franka robot."""

import torch

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
import isaaclab.utils.math as math_utils
from isaaclab.sim import RenderCfg, SimulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass


SOURCE_FLUID_CENTER_OFFSET = (0.0, 0.0, 0.01)
"""Fluid cuboid center relative to the source beaker root pose."""


def _resolve_env_ids(env, env_ids):
    """Return reset env ids as a tensor on the environment device."""
    if env_ids is None:
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    if not isinstance(env_ids, torch.Tensor):
        return torch.as_tensor(env_ids, dtype=torch.long, device=env.device).flatten()
    return env_ids.to(device=env.device, dtype=torch.long).flatten()


def reset_source_fluid_to_beaker(env, env_ids):
    """Re-anchor the source fluid inside the randomized source beaker."""
    scene_keys = set(env.scene.keys())
    if "source_fluid" not in env.particle_systems or "source_beaker" not in scene_keys:
        return

    env_ids = _resolve_env_ids(env, env_ids)

    fluid_cfg = env.cfg.particle_systems["source_fluid"]
    volume = torch.tensor(fluid_cfg.volume, dtype=torch.float32, device=env.device)
    center_offset = torch.tensor(SOURCE_FLUID_CENTER_OFFSET, dtype=torch.float32, device=env.device)

    beaker_positions = env.scene["source_beaker"].data.root_pos_w[env_ids]
    lower_positions = beaker_positions + center_offset - (volume / 2.0)
    lower_positions_list = [tuple(pos.tolist()) for pos in lower_positions]

    env.particle_systems["source_fluid"].reset(env_ids=env_ids, pos=lower_positions_list)


def reset_randomize_pipette_rack(env, env_ids, pose_range, rack_name="pipette_rack"):
    """Randomize the static pipette rack around its configured default pose."""
    if rack_name not in set(env.scene.keys()) or rack_name not in env.cfg.objects:
        return

    env_ids = _resolve_env_ids(env, env_ids)

    rack = env.scene[rack_name]
    rack_cfg = env.cfg.objects[rack_name]
    default_pos = torch.tensor(rack_cfg.init_state.pos, dtype=torch.float32, device=env.device)
    default_quat = torch.tensor(rack_cfg.init_state.rot, dtype=torch.float32, device=env.device)

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, dtype=torch.float32, device=env.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)

    positions = default_pos.unsqueeze(0) + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(
        rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    orientations = math_utils.quat_mul(default_quat.unsqueeze(0).expand(len(env_ids), -1), orientations_delta)

    rack.set_world_poses(positions=positions, orientations=orientations, indices=env_ids)


def reset_pipette_to_pipette_rack(env, env_ids, rack_name="pipette_rack", pipette_name="pipette"):
    """Place the pipette at its configured rack-relative pose after rack randomization."""
    scene_keys = set(env.scene.keys())
    if rack_name not in scene_keys or pipette_name not in scene_keys:
        return
    if rack_name not in env.cfg.objects or pipette_name not in env.cfg.objects:
        return

    env_ids = _resolve_env_ids(env, env_ids)

    rack = env.scene[rack_name]
    pipette = env.scene[pipette_name]
    rack_cfg = env.cfg.objects[rack_name]
    pipette_cfg = env.cfg.objects[pipette_name]

    default_rack_pos = torch.tensor(rack_cfg.init_state.pos, dtype=torch.float32, device=env.device)
    default_rack_quat = torch.tensor(rack_cfg.init_state.rot, dtype=torch.float32, device=env.device)
    default_pipette_pos = torch.tensor(pipette_cfg.init_state.pos, dtype=torch.float32, device=env.device)
    default_pipette_quat = torch.tensor(pipette_cfg.init_state.rot, dtype=torch.float32, device=env.device)

    relative_pos = math_utils.quat_apply_inverse(default_rack_quat, default_pipette_pos - default_rack_pos)
    relative_quat = math_utils.quat_mul(math_utils.quat_inv(default_rack_quat), default_pipette_quat)

    rack_positions, rack_orientations = rack.get_world_poses(indices=env_ids)
    pipette_positions, pipette_orientations = math_utils.combine_frame_transforms(
        rack_positions,
        rack_orientations,
        relative_pos.unsqueeze(0).expand(len(env_ids), -1),
        relative_quat.unsqueeze(0).expand(len(env_ids), -1),
    )

    pipette.write_root_pose_to_sim(
        torch.cat([pipette_positions, pipette_orientations], dim=-1),
        env_ids=env_ids,
    )
    zero_velocity = torch.zeros((len(env_ids), 6), dtype=torch.float32, device=env.device)
    pipette.write_root_velocity_to_sim(zero_velocity, env_ids)


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

    randomize_pipette_rack = EventTerm(
        func=reset_randomize_pipette_rack,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
            },
        },
    )

    sync_pipette_to_pipette_rack = EventTerm(
        func=reset_pipette_to_pipette_rack,
        mode="reset",
    )

    sync_source_fluid_to_beaker = EventTerm(
        func=reset_source_fluid_to_beaker,
        mode="reset",
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
        wrist_rgb = ObsTerm(
            func=mdp.camera_rgb,
            params={"sensor_name": "wrist_camera"},
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
        "pipette": PIPETTE_1ML_INST_CFG(pos=(0.55, -0.2, 0.05)),
        "pipette_rack": PIPETTE_RACK_CFG(pos=(0.55, -0.2, 0.0)),
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
            pos=(0.6, 0.2, 0.01),          # centered inside the source beaker
            volume=(0.02, 0.02, 0.03),      # ~96 mL block of fluid
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
        "wrist_camera": TiledCameraCfg(
            prim_path="/World/envs/env_.*/Articulations_robot/panda_hand/WristCamera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.04, 0.0, 0.06),
                rot=(-0.6963642, -0.1227878, 0.1227878, 0.6963642),  # euler [-90, -20, 0]
                convention="ros",
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=12.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 20.0),
            ),
            width=224,
            height=224,
        ),
    }

    gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]

    # Re-render after resets so the camera sees the freshly-reset scene
    rerender_on_reset = True
    num_rerenders_on_reset: int = 1   # or 0 if you don't need extra renders after reset

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
