"""Workflow execution helpers for dataset generation.

This module wraps MATteRIX state-machine workflow execution so callers can
request one action per env step while keeping episode code compact.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import Any

import torch

from matterix_sm import StateMachine


def _workflow_to_actions(workflow_value: Any) -> list[Any]:
    """Normalize workflow config value into a list of actions/configs."""
    if isinstance(workflow_value, dict):
        return list(workflow_value.get("actions", []))
    return [workflow_value]


def _extract_action_tensor(action_output: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
    """Convert StateMachine output to a single action tensor."""
    if isinstance(action_output, dict):
        if len(action_output) != 1:
            raise ValueError(
                f"Multi-agent workflows are not supported in dataset generation (got {len(action_output)} agents)."
            )
        return next(iter(action_output.values()))
    return action_output


@dataclass
class WorkflowExecutor:
    """State-machine wrapper for one environment/workflow pair."""

    env: Any
    workflow_name: str
    workflow_value: Any
    suppress_output: bool = False

    @contextlib.contextmanager
    def _mute_state_machine_output(self):
        """Silence StateMachine's direct stdout/stderr prints when requested."""
        if not self.suppress_output:
            yield
            return

        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                yield

    def __post_init__(self) -> None:
        actions = _workflow_to_actions(self.workflow_value)
        self.state_machine = StateMachine(
            num_envs=self.env.num_envs,
            dt=self.env.step_dt,
            device=self.env.device,
        )
        with self._mute_state_machine_output():
            self.state_machine.set_action_sequence(actions)

    def reset(self) -> None:
        """Reset state-machine episode state."""
        self.state_machine.reset()

    def step(self, obs: dict[str, Any]) -> torch.Tensor:
        """Compute the next action tensor on the environment device."""
        with self._mute_state_machine_output():
            action_output = self.state_machine.step(obs)
        action_tensor = _extract_action_tensor(action_output)
        return action_tensor.to(self.env.device)

    def is_done(self, env_index: int = 0) -> bool:
        """Whether a specific environment has finished (success or failure)."""
        done = self.state_machine.action_sequence_success | self.state_machine.action_sequence_failure
        return bool(done[env_index].item())

    def succeeded(self, env_index: int = 0) -> bool:
        """Whether a specific environment finished successfully."""
        return bool(self.state_machine.action_sequence_success[env_index].item())

    def current_action_index(self, env_index: int = 0) -> int:
        """Current primitive action index for a specific environment."""
        return int(self.state_machine.current_action_idx[env_index].item())


def create_workflow_executor(
    env: Any,
    env_cfg: Any,
    preferred_workflow_name: str | None = None,
    suppress_output: bool = False,
) -> WorkflowExecutor:
    """Create a workflow executor from an environment cfg.

    Args:
        env: Created gym/Isaac environment instance.
        env_cfg: Parsed environment config containing ``workflows``.
        preferred_workflow_name: Optional explicit workflow name.

    Returns:
        Configured ``WorkflowExecutor``.
    """
    workflows = getattr(env_cfg, "workflows", None)
    if not workflows:
        raise ValueError("Environment has no workflows configured.")

    workflow_name = preferred_workflow_name
    if workflow_name is None:
        workflow_name = next(iter(workflows.keys()))

    if workflow_name not in workflows:
        available = list(workflows.keys())
        raise ValueError(
            f"Workflow '{workflow_name}' not found. Available workflows: {available}"
        )

    return WorkflowExecutor(
        env=env,
        workflow_name=workflow_name,
        workflow_value=workflows[workflow_name],
        suppress_output=suppress_output,
    )
