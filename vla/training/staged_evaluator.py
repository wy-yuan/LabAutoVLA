"""Run a scripted workflow prefix before handing the same scene to a policy."""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch

log = logging.getLogger(__name__)

# Names refer to completed primitives, including the lift after picking/dipping.
PIPETTING_STAGES = (
    "pre_grasp", "open_gripper", "grasp", "close_gripper", "picking",
    "source_approach", "source_dip", "source",
    "target_approach", "target_dip", "target",
)
_PRIMITIVE_TYPES = (
    "MoveToFrame", "OpenGripper", "MoveToFrame", "CloseGripper", "MoveToFrame",
    "MoveToFrame", "MoveRelative", "MoveRelative",
    "MoveToFrame", "MoveRelative", "MoveRelative",
)


def resolve_handoff(executor: Any, after_stage: str, after_action: int | None) -> int:
    """Return the number of primitives to complete; numeric indices are zero-based."""
    actions = executor.state_machine.actions
    if after_action is not None:
        if isinstance(after_action, bool) or not isinstance(after_action, int):
            raise ValueError("handoff.after_action must be an integer or null")
        if not 0 <= after_action < len(actions):
            raise ValueError(f"handoff.after_action must be in [0, {len(actions) - 1}]")
        return after_action + 1
    if after_stage == "reset":
        return 0

    counts = {"picking_pipette": 5, "pipette_liquid": 11}
    if executor.workflow_name not in counts:
        raise ValueError("Named stages support picking_pipette/pipette_liquid; use handoff.after_action")
    count = counts[executor.workflow_name]
    stages = PIPETTING_STAGES[:count]
    if after_stage not in stages:
        raise ValueError(f"Unknown stage {after_stage!r}; choose reset or one of {stages}")
    actual = tuple(type(action).__name__ for action in actions)
    if actual != _PRIMITIVE_TYPES[:count]:
        raise ValueError(
            "Workflow primitives changed; update named stage mappings or use handoff.after_action. "
            f"Actual sequence: {actual}"
        )
    return stages.index(after_stage) + 1


def run_staged_episode(
    env: Any,
    policy: Any,
    executor: Any,
    *,
    completed_actions: int,
    scripted_max_steps: int,
    max_steps: int,
    record_frame: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Reset once, run the prefix in native action units, then run the VLA suffix.

    ``max_steps`` counts policy steps only. Environment terminations may reset the
    scene internally, so neither phase continues after a terminal transition.
    """
    if env.unwrapped.num_envs != 1:
        raise ValueError("Staged evaluation requires num_envs=1")
    if scripted_max_steps < 0 or max_steps <= 0:
        raise ValueError("scripted_max_steps must be nonnegative and max_steps must be positive")
    if not 0 <= completed_actions <= len(executor.state_machine.actions):
        raise ValueError("completed_actions is outside the workflow")

    policy.eval()
    policy.reset()
    raw_env = env.unwrapped
    raw_obs, _ = raw_env.reset()
    reset_state_history = getattr(env, "reset_state_history", None)
    if callable(reset_state_history):
        reset_state_history(raw_obs)
    executor.reset()
    result = {
        "handoff_reached": False,
        "scripted_steps": 0,
        "policy_steps": 0,
        "handoff_frame": None,
        "handoff_state": None,
        "next_action_index": 0,
        "terminated": False,
        "truncated": False,
        "stop_reason": "scripted_step_limit",
    }

    def record() -> None:
        if record_frame is not None:
            record_frame()

    def ended(terminated: Any, truncated: Any, phase: str) -> bool:
        result["terminated"] = bool(torch.as_tensor(terminated).any())
        result["truncated"] = bool(torch.as_tensor(truncated).any())
        if result["terminated"] or result["truncated"]:
            suffix = "terminated" if result["terminated"] else "truncated"
            result["stop_reason"] = f"{phase}_{suffix}"
            return True
        return False

    record()
    # Isaac Lab caches tensors during step(); they must remain writable at reset.
    with torch.no_grad():
        while completed_actions:
            # step() checks completion against the current observation and returns
            # the OLD primitive's command. Stop before starting the next primitive.
            action = executor.step(raw_obs)
            result["next_action_index"] = executor.current_action_index()
            if executor.is_done() and not executor.succeeded():
                result["stop_reason"] = "scripted_action_failed"
                return result
            if result["next_action_index"] >= completed_actions:
                break
            if result["scripted_steps"] >= scripted_max_steps:
                return result
            if not torch.isfinite(action).all():
                raise RuntimeError("Scripted workflow produced a non-finite action")
            # Workflow gripper commands are already in meters, unlike VLA actions.
            raw_obs, _, terminated, truncated, _ = raw_env.step(action)
            observe_raw_state = getattr(env, "observe_raw_state", None)
            if callable(observe_raw_state):
                observe_raw_state(raw_obs)
            result["scripted_steps"] += 1
            if ended(terminated, truncated, "scripted"):
                return result
            record()

        # Keep the latest observation and physical state; discard only policy memory.
        policy.reset()
        adapt_current = getattr(env, "adapt_current", None)
        obs = adapt_current(raw_obs) if callable(adapt_current) else env._adapt(raw_obs)
        result["handoff_reached"] = True
        result["handoff_frame"] = result["scripted_steps"] if record_frame is not None else None
        result["handoff_state"] = obs["state"].detach().cpu().tolist()
        log.info(
            "Handoff after %d scripted steps; next primitive=%d",
            result["scripted_steps"], result["next_action_index"],
        )
        for _ in range(max_steps):
            with torch.inference_mode():
                action = policy.predict_action(
                    images=obs["images"], state=obs["state"], task=obs["task"]
                )
            if not torch.isfinite(action).all():
                raise RuntimeError("Policy produced a non-finite action")
            obs, _, terminated, truncated, _ = env.step(action)
            result["policy_steps"] += 1
            if ended(terminated, truncated, "policy"):
                return result
            record()

    result["stop_reason"] = "policy_step_limit"
    return result
