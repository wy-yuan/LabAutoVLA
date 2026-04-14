# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for pipette and pipette rack.

Frame layout for the pipette (relative to pipette body origin at grip centre):
    pre_grasp  : 8 cm above the grip zone — approach before grasping
    grasp      : grip zone (origin of the pipette body)
    post_grasp : 12 cm above the grip zone — safe lift height after pick-up
    tip_pre    : 2 cm above the pipette tip — approach before dipping into liquid
    tip        : at the pipette tip — used to calibrate dip depth

The following configurations are available:

* :obj:`PIPETTE_1ML_INST_CFG`: Instantiated 1 mL single-channel pipette
* :obj:`PIPETTE_RACK_CFG`: Pipette rack (static object)
"""

from matterix_assets import MATTERIX_ASSETS_DATA_DIR

from isaaclab.sensors import OffsetCfg
from isaaclab.utils import configclass

from ..matterix_rigid_object import MatterixRigidObjectCfg
from ..matterix_static_object import MatterixStaticObjectCfg

##
# Configuration
##

default_rigid_prim_path = MatterixRigidObjectCfg().prim_path + "_Labware"
default_static_prim_path = MatterixStaticObjectCfg().prim_path + "_Labware"


@configclass
class PIPETTE_1ML_INST_CFG(MatterixRigidObjectCfg):
    """Properties for the 1 mL pipette in the scene.

    Manipulation frames are defined in the pipette body frame (origin at grip centre):
    - pre_grasp / grasp / post_grasp: for the robot to pick up the pipette
    - tip_pre / tip: for positioning the pipette over a liquid container

    NOTE: The USD uses metersPerUnit = 0.1 (decimetres) and upAxis = "Z",
    so scale = 0.1 compensates for the unit mismatch.
    """

    prim_path = default_rigid_prim_path
    usd_path = f"{MATTERIX_ASSETS_DATA_DIR}/labware/pipette/Pipette_vertical.usda" 

    scale = (0.1, 0.1, 0.1)
    rot = (0.707, 0.0, 0.0, -0.707)  # -90° around Z-axis
    mass = 0.05  # kg (~50 g for a typical hand pipette)
    activate_contact_sensors = False

    # frames = {
    #     # Frames for picking up the pipette from its rack
    #     "pre_grasp": OffsetCfg(pos=(0.0, -0.1, 0.2), rot=(0.5, 0.5, -0.5, 0.5)), # Top-down approach
    #     "grasp": OffsetCfg(pos=(0.0, -0.01, 0.2), rot=(0.5, 0.5, -0.5, 0.5)),         # grip centre (body origin)
    #     "post_grasp": OffsetCfg(pos=(0.0, -0.01, 0.1), rot=(0.5, 0.5, -0.5, 0.5)), # initial test - Fingers point forward: (0.5, 0.5, -0.5, 0.5)
    #     # Frames for dipping the tip into liquid containers
    #     # Values assume the pipette tip is ~18 cm below the grip centre
    #     "tip_pre": (0.0, 0.0, -0.16),     # 2 cm above the tip
    #     "tip": (0.0, 0.0, -0.18),         # at the pipette tip
    # }
    frames = {
        # Frames for picking up the pipette from its rack
        "pre_grasp": OffsetCfg(pos=(0.0, 0.0, 0.3), rot=(0.707, 0.0, 0.0, 0.707)), # Top-down approach
        "grasp": OffsetCfg(pos=(0.0, 0.0, 0.08), rot=(0.707, 0.0, 0.0, 0.707)),         # grip centre (body origin)
        "post_grasp": OffsetCfg(pos=(0.0, 0.0, 0.3), rot=(0.707, 0.0, 0.0, 0.707)), # initial test - Fingers point forward: (0.5, 0.5, -0.5, 0.5)
        "tip_pre": (0.0, 0.0, -0.16),     # 2 cm above the tip
        "tip": (0.0, 0.0, -0.18),         # at the pipette tip
    }

    semantic_tags = [("class", "pipette")]


@configclass
class PIPETTE_RACK_CFG(MatterixStaticObjectCfg):
    """Pipette rack — static collision mesh that holds pipettes on the table."""

    prim_path = default_static_prim_path
    usd_path = f"{MATTERIX_ASSETS_DATA_DIR}/labware/pipetteRack/PipetteHolder.usda"

    scale = (0.101, 0.101, 0.101)  # USD uses metersPerUnit=0.1, scale down 10x
    rot = (0.707, 0.0, 0.0, -0.707)  # -90° around Z-axis
    semantic_tags = [("class", "pipette_rack")]
