"""Custom gripper action terms for Matterix robot assets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass


class MirroredJointPositionAction(ActionTerm):
    """Single scalar joint-position command mirrored to all matching joints."""

    cfg: MirroredJointPositionActionCfg

    def __init__(self, cfg: MirroredJointPositionActionCfg, env) -> None:
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        if self._num_joints == 0:
            raise ValueError(f"No joints matched gripper action patterns: {self.cfg.joint_names}")

        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._scale = float(self.cfg.scale)
        self._offset = float(self.cfg.offset)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        command = actions[:, 0:1] * self._scale + self._offset
        if self.cfg.command_clip is not None:
            command = command.clamp(min=self.cfg.command_clip[0], max=self.cfg.command_clip[1])
        self._processed_actions[:] = command.expand(-1, self._num_joints)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


@configclass
class MirroredJointPositionActionCfg(ActionTermCfg):
    """Configuration for a one-dimensional joint-position command mirrored to multiple joints."""

    class_type: type[ActionTerm] = MirroredJointPositionAction
    joint_names: list[str] = MISSING
    scale: float = 1.0
    offset: float = 0.0
    command_clip: tuple[float, float] | None = None
