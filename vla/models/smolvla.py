"""SmolVLA wrapper.

Wraps LeRobot's ``SmolVLAPolicy`` so it conforms to :class:`BaseVLA`.
SmolVLA (HuggingFace ``lerobot/smolvla_base``) is a compact VLA that
takes:
    * one or more RGB images (224x224, float [0,1], BCHW)
    * proprioceptive ``observation.state`` (flat float vector)
    * a natural-language ``task`` string
and outputs an action chunk via flow matching.

Why a thin wrapper rather than subclassing ``SmolVLAPolicy`` directly?
    * Keeps our training code agnostic of LeRobot's evolving internal API.
    * Lets us normalise batch keys (Matterix/LeRobot drift) in one place.
    * Makes swapping in OpenVLA / pi_0 / GR00T a matter of adding a
      sibling file — no call-site changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .base_vla import BaseVLA, VLAOutput
from .registry import register_vla

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quaternion helpers — [w, x, y, z] convention, matching Isaac Lab / LeRobot.
# Used for relative-action conversion when use_relative_actions=True.
# State layout: [ee_pos(3), ee_quat(4), gripper(2)]
# Action layouts:
#   8D: [pos(3), quat(4), gripper_open(1)]
#   9D: [pos(3), quat(4), gripper_pos(2)]
# ---------------------------------------------------------------------------

def _quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Conjugate (== inverse for unit quaternions), [..., 4] [w,x,y,z]."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize quaternions and canonicalize sign for smoother targets."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
    return torch.where(q[..., :1] < 0, -q, q)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 * q2, [..., 4] [w,x,y,z]."""
    w1, x1, y1, z1 = q1[..., 0:1], q1[..., 1:2], q1[..., 2:3], q1[..., 3:4]
    w2, x2, y2, z2 = q2[..., 0:1], q2[..., 1:2], q2[..., 2:3], q2[..., 3:4]
    return torch.cat([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert [..., 4] quaternions [w,x,y,z] to [..., 3, 3] matrices."""
    q = _quat_normalize(q)
    w, x, y, z = q.unbind(dim=-1)
    two_s = 2.0
    return torch.stack(
        [
            1 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1 - two_s * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def _matrix_to_quat(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert [..., 3, 3] rotation matrices to quaternions [w,x,y,z]."""
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    qw = 0.5 * torch.sqrt((1.0 + m00 + m11 + m22).clamp_min(eps))
    qx = torch.copysign(
        0.5 * torch.sqrt((1.0 + m00 - m11 - m22).clamp_min(eps)),
        matrix[..., 2, 1] - matrix[..., 1, 2],
    )
    qy = torch.copysign(
        0.5 * torch.sqrt((1.0 - m00 + m11 - m22).clamp_min(eps)),
        matrix[..., 0, 2] - matrix[..., 2, 0],
    )
    qz = torch.copysign(
        0.5 * torch.sqrt((1.0 - m00 - m11 + m22).clamp_min(eps)),
        matrix[..., 1, 0] - matrix[..., 0, 1],
    )
    return _quat_normalize(torch.stack([qw, qx, qy, qz], dim=-1))


def quat_to_rotation_6d(q: torch.Tensor) -> torch.Tensor:
    """Convert [..., 4] quaternions to Zhou 6D rotation columns."""
    matrix = _quat_to_matrix(q)
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)


def rotation_6d_to_quat(d6: torch.Tensor) -> torch.Tensor:
    """Convert Zhou 6D rotation columns to [..., 4] quaternions [w,x,y,z]."""
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-8)
    b2 = torch.nn.functional.normalize(
        a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1,
        dim=-1,
        eps=1e-8,
    )
    b3 = torch.cross(b1, b2, dim=-1)
    matrix = torch.stack([b1, b2, b3], dim=-1)
    return _matrix_to_quat(matrix)


def action_quat_to_rotation_6d(action: torch.Tensor) -> torch.Tensor:
    """Map raw pose actions [pos(3), quat(4), gripper...] to learned 6D actions."""
    if action.shape[-1] < 7:
        raise ValueError(f"Quaternion action must have at least 7 dims, got {action.shape[-1]}")
    return torch.cat(
        [action[..., :3], quat_to_rotation_6d(action[..., 3:7]), action[..., 7:]],
        dim=-1,
    )


def action_rotation_6d_to_quat(action: torch.Tensor) -> torch.Tensor:
    """Map learned 6D actions [pos(3), rot6(6), gripper...] back to raw pose actions."""
    if action.shape[-1] < 9:
        raise ValueError(f"6D rotation action must have at least 9 dims, got {action.shape[-1]}")
    return torch.cat(
        [action[..., :3], rotation_6d_to_quat(action[..., 3:9]), action[..., 9:]],
        dim=-1,
    )


def learned_action_dim_from_raw(raw_action_dim: int, *, use_rotation_6d: bool = True) -> int:
    """Return policy output dim for a raw [pos, quat, gripper...] action dim."""
    return int(raw_action_dim) + 2 if use_rotation_6d else int(raw_action_dim)


def _same_hemisphere(q: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Flip ``q`` if needed so it represents rotation near ``reference``."""
    dot = (q * reference).sum(dim=-1, keepdim=True)
    return torch.where(dot < 0, -q, q)


def _gripper_open_from_state(state: torch.Tensor) -> torch.Tensor:
    """Convert two observed finger positions to a single [0, 1] opening."""
    grip = state[..., -2:] if state.shape[-1] >= 2 else state[..., -1:]
    if grip.shape[-1] == 1:
        return grip.clamp(0.0, 1.0)
    return grip.abs().clamp(0.0, 0.04).mean(dim=-1, keepdim=True) / 0.04


def _state_to_action_pose(state: torch.Tensor) -> torch.Tensor:
    """Return current EE pose in the 8D base-frame action layout.

    State layout:
      [ee_pos(3), ee_quat(4), gripper_pos(2)]

    The robot base is fixed for the current tasks, so the stored EE world pose
    is already in the action frame.
    """
    if state.shape[-1] != 9:
        raise ValueError(
            "State must be compact 9D [ee_pos(3), ee_quat(4), gripper_pos(2)], "
            f"got shape {tuple(state.shape)}"
        )
    return torch.cat(
        [state[..., :3], _quat_normalize(state[..., 3:7]), _gripper_open_from_state(state)],
        dim=-1,
    )


def actions_to_relative(actions: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """Convert absolute base-frame EE-pose actions to relative actions.

    All chunk steps are expressed relative to the state at t0 (the current
    observation), so the model learns offsets rather than absolute targets.

    args:
        actions: (B, T, 8/9) or (B, 8/9) — [pos(3), quat(4), gripper...]
        state:   (B, 9)               — [pos(3), quat(4), gripper(2)]

    returns: same shape as *actions*
        [pos_delta(3), quat_rel(4), gripper_abs(...)]
    """
    squeeze = actions.ndim == 2
    if squeeze:
        actions = actions.unsqueeze(1)  # (B, 1, 8)

    state_pose = _state_to_action_pose(state)
    state_pos = state_pose[:, :3].unsqueeze(1)
    state_quat = state_pose[:, 3:7].unsqueeze(1)
    action_quat = _same_hemisphere(
        _quat_normalize(actions[..., 3:7]),
        state_quat.expand_as(actions[..., 3:7]),
    )

    rel_pos = actions[..., :3] - state_pos
    # q_action = q_state * q_rel, so q_rel = q_state^{-1} * q_action.
    rel_quat = _quat_normalize(
        _quat_mul(_quat_inv(state_quat).expand_as(action_quat), action_quat)
    )
    gripper = actions[..., 7:]  # gripper is an absolute command/position

    if actions.shape[-1] == 8:
        gripper = gripper.clamp(0.0, 1.0)
    result = torch.cat([rel_pos, rel_quat, gripper], dim=-1)
    return result.squeeze(1) if squeeze else result


def actions_to_absolute(actions: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """Invert :func:`actions_to_relative` to recover base-frame actions.

    args:
        actions: (B, T, 8/9) or (B, 8/9) — [pos_delta(3), quat_rel(4), gripper...]
        state:   (B, 9)               — [pos(3), quat(4), gripper(2)]

    returns: same shape as *actions*, base-frame
    """
    squeeze = actions.ndim == 2
    if squeeze:
        actions = actions.unsqueeze(1)

    state_pose = _state_to_action_pose(state)
    state_pos = state_pose[:, :3].unsqueeze(1)
    state_quat = state_pose[:, 3:7].unsqueeze(1)
    rel_quat = _quat_normalize(actions[..., 3:7])

    abs_pos = actions[..., :3] + state_pos
    # q_action = q_state * q_rel.
    abs_quat = _quat_normalize(_quat_mul(state_quat.expand_as(rel_quat), rel_quat))
    gripper = actions[..., 7:]

    if actions.shape[-1] == 8:
        gripper = gripper.clamp(0.0, 1.0)
    result = torch.cat([abs_pos, abs_quat, gripper], dim=-1)
    return result.squeeze(1) if squeeze else result


def action_to_env_action(action: torch.Tensor) -> torch.Tensor:
    """Convert base-frame policy action to the Matterix env command."""
    if action.shape[-1] in (10, 11):
        action = action_rotation_6d_to_quat(action)
    if action.shape[-1] != 8:
        return action
    processed = action.clone()
    processed[..., 3:7] = _quat_normalize(processed[..., 3:7])
    gripper_open = processed[..., 7:8].clamp(0.0, 1.0)
    processed[..., 7:8] = torch.where(
        gripper_open >= 0.5,
        torch.ones_like(gripper_open),
        -torch.ones_like(gripper_open),
    )
    return processed


try:
    from lerobot.configs.types import PipelineFeatureType, PolicyFeature
    from lerobot.processor import ActionProcessorStep, ProcessorStep, ProcessorStepRegistry
    from lerobot.processor.core import TransitionKey
    from lerobot.utils.constants import OBS_STATE

    @dataclass
    @ProcessorStepRegistry.register(name="labauto_relative_ee_action")
    class RelativeEEActionProcessorStep(ProcessorStep):
        """Convert absolute EE-pose actions to LabAuto's relative target space."""

        state_key: str = OBS_STATE

        def __call__(self, transition):
            self._current_transition = transition.copy()
            new_transition = self._current_transition
            action = new_transition.get(TransitionKey.ACTION)
            if action is None:
                return new_transition
            if not isinstance(action, torch.Tensor):
                raise ValueError(f"Relative EE action expects a tensor action, got {type(action)}")

            observation = new_transition.get(TransitionKey.OBSERVATION) or {}
            state = observation.get(self.state_key)
            if state is None:
                raise ValueError(
                    f"Relative action preprocessing requires '{self.state_key}' in the observation."
                )
            state = torch.as_tensor(state, device=action.device, dtype=action.dtype)
            if state.ndim == 3:
                state = state[:, -1]
            new_transition[TransitionKey.ACTION] = actions_to_relative(action, state)
            return new_transition

        def transform_features(
            self,
            features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
        ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
            return features

    @dataclass
    @ProcessorStepRegistry.register(name="labauto_quat_action_to_rotation_6d")
    class QuaternionActionToRotation6DProcessorStep(ProcessorStep):
        """Convert action quaternions to Zhou 6D before LeRobot normalization."""

        def __call__(self, transition):
            self._current_transition = transition.copy()
            new_transition = self._current_transition
            action = new_transition.get(TransitionKey.ACTION)
            if action is None:
                return new_transition
            new_transition[TransitionKey.ACTION] = self.action(action)
            return new_transition

        def action(self, action):
            if not isinstance(action, torch.Tensor):
                raise ValueError(f"6D rotation preprocessing expects a tensor action, got {type(action)}")
            return action_quat_to_rotation_6d(action)

        def transform_features(
            self,
            features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
        ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
            return features

except (ImportError, ValueError):  # pragma: no cover - LeRobot may be absent during lightweight tooling.
    RelativeEEActionProcessorStep = None  # type: ignore[assignment]
    QuaternionActionToRotation6DProcessorStep = None  # type: ignore[assignment]


@register_vla("smolvla")
class SmolVLA(BaseVLA):
    """Thin adapter around ``lerobot.common.policies.smolvla.SmolVLAPolicy``."""

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        image_keys: Sequence[str],
        pretrained_name_or_path: str = "lerobot/smolvla_base",
        # Optional overrides — left None to defer to the upstream defaults.
        chunk_size: int | None = None,
        n_action_steps: int | None = None,
        num_steps: int | None = None,
        use_cache: bool | None = None,
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = False,
        load_pretrained: bool = True,
        load_vlm_weights: bool = False,
        use_relative_actions: bool = False,
        use_rotation_6d: bool = True,
        step0_loss_weight: float = 1.0,
        device: str | torch.device = "cuda",
    ):
        super().__init__(action_dim=action_dim, state_dim=state_dim, image_keys=image_keys)
        self._device = torch.device(device)
        self.use_relative_actions = use_relative_actions
        self.use_rotation_6d = use_rotation_6d
        self.step0_loss_weight = step0_loss_weight
        self._action_norm_mode = "relative" if self.use_relative_actions else "absolute"
        self._processor_stats: dict[str, dict[str, Any]] = {}
        self.preprocessor: Any | None = None
        self.postprocessor: Any | None = None
        # Tracks which step in the current action chunk we are (for inference).
        self._rel_base_state: torch.Tensor | None = None
        self._rel_step_in_chunk: int = 0

        # Defer LeRobot import so the package is optional at collection /
        # linting time and so import errors surface with useful context.
        try:
            from lerobot.configs.types import FeatureType, PolicyFeature
            from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SmolVLA requires the `lerobot` package (pip install lerobot). "
                "See requirements.txt."
            ) from exc

        # -- Build config -------------------------------------------------
        # SmolVLAConfig exposes features for input/output normalisation.
        # We construct a minimal config here; dataset stats are attached
        # later by the training loop (see bc_train.py).
        cfg_kwargs: dict[str, Any] = {}
        if chunk_size is not None:
            cfg_kwargs["chunk_size"] = chunk_size
        if n_action_steps is not None:
            cfg_kwargs["n_action_steps"] = n_action_steps
        if num_steps is not None:
            cfg_kwargs["num_steps"] = num_steps
        if use_cache is not None:
            cfg_kwargs["use_cache"] = use_cache
        cfg_kwargs["freeze_vision_encoder"] = freeze_vision_encoder
        cfg_kwargs["train_expert_only"] = train_expert_only
        cfg_kwargs["load_vlm_weights"] = load_vlm_weights
        cfg_kwargs["device"] = str(self._device)
        cfg_kwargs["input_features"] = {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            **{
                f"{OBS_IMAGES}.{image_key}": PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, 224, 224),
                )
                for image_key in image_keys
            },
        }
        cfg_kwargs["output_features"] = {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))
        }

        self.cfg = SmolVLAConfig(**cfg_kwargs)

        # -- Build policy -------------------------------------------------
        if load_pretrained:
            _ckpt = Path(pretrained_name_or_path)
            if _ckpt.suffix == ".safetensors":
                pretrained_name_or_path = str(_ckpt.parent)
            self.policy = SmolVLAPolicy.from_pretrained(pretrained_name_or_path, config=self.cfg)
        else:
            self.policy = SmolVLAPolicy(self.cfg)

        self.to(self._device)
        if not self._try_load_processors_from_checkpoint(pretrained_name_or_path):
            self.configure_processors()

    # ------------------------------------------------------------------
    # Contract
    # ------------------------------------------------------------------
    def compute_loss(self, batch: Mapping[str, Any]) -> VLAOutput:
        """Delegate to the underlying policy's forward-with-loss path."""
        batch = self._move_to_device(batch)
        # LeRobot policies return either a dict (preferred) or a tuple
        # (loss, info). We normalise.
        result = self.policy.forward(batch)
        if isinstance(result, tuple):
            loss, info = result[0], result[1] if len(result) > 1 else {}
        elif isinstance(result, dict):
            loss = result.get("loss")
            info = {k: v for k, v in result.items() if k != "loss"}
        else:
            loss, info = result, {}

        # Extra weight on step-0: second forward pass with only step-0 unmasked.
        # Adds (step0_loss_weight - 1) * L_step0 so the gradient from step-0
        # is amplified relative to later chunk steps.
        if self.step0_loss_weight != 1.0 and loss is not None:
            step0_batch = self._make_step0_only_batch(batch)
            r0 = self.policy.forward(step0_batch)
            if isinstance(r0, tuple):
                loss0 = r0[0]
            elif isinstance(r0, dict):
                loss0 = r0.get("loss")
            else:
                loss0 = r0
            if loss0 is not None:
                loss = loss + (self.step0_loss_weight - 1.0) * loss0

        return VLAOutput(actions=None, loss=loss, aux=info)

    @torch.inference_mode()
    def predict_action(
        self,
        images: Mapping[str, torch.Tensor],
        state: torch.Tensor,
        task: Sequence[str],
    ) -> torch.Tensor:
        """Rollout helper — returns a single-step action ``(B, A)``."""
        state_dev = state.to(self._device)
        current_state = state_dev[:, -1].contiguous() if state_dev.ndim == 3 else state_dev

        if self.use_relative_actions:
            n_steps = getattr(self.cfg, "n_action_steps", 1) or 1
            if self._rel_step_in_chunk == 0:
                # Capture state at the start of each planning chunk so all
                # actions in the chunk are converted relative to the same origin.
                self._rel_base_state = current_state.clone()

        observation: dict[str, Any] = {
            "observation.state": current_state,
            "task": list(task),
        }
        for k in self.image_keys:
            if k not in images:
                raise KeyError(f"SmolVLA expects image key '{k}' — got {list(images)}")
            image = images[k].to(self._device)
            current_image = image[:, -1].contiguous() if image.ndim == 5 else image
            observation[f"observation.images.{k}"] = current_image

        observation = self.preprocess_batch(observation)

        # ``select_action`` streams one action per call out of the internal
        # chunk buffer; call ``reset()`` between episodes (see BaseVLA.reset).
        action = self.policy.select_action(observation)
        action = self.postprocess_action(action)

        if self.use_relative_actions and self._rel_base_state is not None:
            action = actions_to_absolute(action, self._rel_base_state.to(action.device))
            n_steps = getattr(self.cfg, "n_action_steps", 1) or 1
            self._rel_step_in_chunk = (self._rel_step_in_chunk + 1) % n_steps

        return action  # (B, action_dim)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear the internal action-chunk queue between episodes."""
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self._rel_base_state = None
        self._rel_step_in_chunk = 0

    # ------------------------------------------------------------------
    # Relative-action helpers
    # ------------------------------------------------------------------
    def _make_step0_only_batch(self, batch: dict) -> dict:
        """Return a copy of batch where only step-0 is unmasked in the action chunk."""
        step0 = dict(batch)
        action = batch.get("action")
        if action is None or action.ndim < 3:
            return step0
        B, T = action.shape[:2]
        # All steps padded except step-0
        pad = torch.ones(B, T, dtype=torch.bool, device=action.device)
        pad[:, 0] = False
        step0["actions_id_pad"] = pad
        if "action_is_pad" in step0:
            step0["action_is_pad"] = pad
        return step0

    @staticmethod
    def _copy_stats(stats: Mapping[str, Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
        copied: dict[str, dict[str, Any]] = {}
        for key, feature_stats in (stats or {}).items():
            copied[key] = {}
            for stat_name, value in feature_stats.items():
                if isinstance(value, torch.Tensor):
                    copied[key][stat_name] = value.detach().cpu().clone()
                elif hasattr(value, "copy"):
                    copied[key][stat_name] = value.copy()
                else:
                    copied[key][stat_name] = value
        return copied

    @staticmethod
    def _action_stats_from_mean_std(mean: torch.Tensor, std: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "mean": mean.detach().float().cpu().clone(),
            "std": std.detach().float().cpu().clamp(min=1e-6).clone(),
        }

    def configure_processors(
        self,
        dataset_stats: Mapping[str, Mapping[str, Any]] | None = None,
        *,
        action_stats: Mapping[str, Any] | None = None,
        action_stats_mode: str | None = None,
    ) -> None:
        """Build LeRobot pre/postprocessors with dataset statistics.

        ``dataset_stats`` should normally be ``LeRobotDataset.meta.stats``.
        For relative-action training, pass the same dataset stats with an
        ``action_stats`` override computed in relative target space.
        """
        try:
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.processor import NormalizerProcessorStep
            from lerobot.utils.constants import ACTION
        except ImportError as exc:  # pragma: no cover
            raise ImportError("SmolVLA processors require the `lerobot` package.") from exc

        stats = self._copy_stats(dataset_stats)
        if action_stats is not None:
            stats[ACTION] = {
                k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                for k, v in action_stats.items()
            }
        if action_stats_mode is not None:
            if action_stats_mode not in ("absolute", "relative"):
                raise ValueError(f"Unsupported action normalization mode: {action_stats_mode!r}")
            self._action_norm_mode = action_stats_mode

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.cfg,
            dataset_stats=stats or None,
        )
        if self.use_relative_actions:
            if RelativeEEActionProcessorStep is None:
                raise ImportError("Relative action preprocessing requires LeRobot processors.")
            if not any(isinstance(step, RelativeEEActionProcessorStep) for step in self.preprocessor.steps):
                steps = list(self.preprocessor.steps)
                insert_at = next(
                    (idx for idx, step in enumerate(steps) if isinstance(step, NormalizerProcessorStep)),
                    len(steps),
                )
                steps.insert(insert_at, RelativeEEActionProcessorStep())
                self.preprocessor.steps = steps

        if self.use_rotation_6d:
            if QuaternionActionToRotation6DProcessorStep is None:
                raise ImportError("6D rotation preprocessing requires LeRobot processors.")
            if not any(
                isinstance(step, QuaternionActionToRotation6DProcessorStep)
                for step in self.preprocessor.steps
            ):
                steps = list(self.preprocessor.steps)
                insert_at = next(
                    (idx for idx, step in enumerate(steps) if isinstance(step, NormalizerProcessorStep)),
                    len(steps),
                )
                steps.insert(insert_at, QuaternionActionToRotation6DProcessorStep())
                self.preprocessor.steps = steps

        self._processor_stats = stats

    def _try_load_processors_from_checkpoint(self, pretrained_name_or_path: str | Path) -> bool:
        """Load saved LeRobot processor pipelines from a local checkpoint, when present."""
        ckpt_dir = Path(pretrained_name_or_path)
        if ckpt_dir.suffix == ".safetensors":
            ckpt_dir = ckpt_dir.parent
        if not ckpt_dir.exists():
            return False

        try:
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.utils.constants import (
                POLICY_POSTPROCESSOR_DEFAULT_NAME,
                POLICY_PREPROCESSOR_DEFAULT_NAME,
            )
        except ImportError:
            return False

        pre_cfg = ckpt_dir / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
        post_cfg = ckpt_dir / f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
        if not pre_cfg.exists() or not post_cfg.exists():
            return False

        try:
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy_cfg=self.cfg,
                pretrained_path=str(ckpt_dir),
                preprocessor_overrides={
                    "device_processor": {"device": str(self._device)},
                },
            )
            self._processor_stats = {}
            log.info("Loaded LeRobot processor pipelines from %s", ckpt_dir)
            return True
        except Exception as exc:
            log.warning("Could not load LeRobot processors from %s: %s", ckpt_dir, exc)
            return False

    def has_processor_stats(self) -> bool:
        """Return whether action normalization stats are available in processors."""
        try:
            from lerobot.utils.constants import ACTION
        except ImportError:
            ACTION = "action"

        if {"mean", "std"}.issubset(self._processor_stats.get(ACTION, {})):
            return True
        for pipeline in (self.preprocessor, self.postprocessor):
            for step in getattr(pipeline, "steps", []):
                stats = getattr(step, "stats", None) or getattr(step, "_tensor_stats", None)
                if isinstance(stats, dict) and {"mean", "std"}.issubset(stats.get(ACTION, {})):
                    return True
        return False

    def has_action_normalizer_stats(self) -> bool:
        """Backward-compatible alias for callers that predate processor pipelines."""
        return self.has_processor_stats()

    def update_action_normalizer_stats(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        mode: str | None = None,
    ) -> None:
        """Backward-compatible action-stat override.

        Prefer ``configure_processors(dataset.meta.stats, action_stats=...)`` so
        state/image stats are preserved. This method keeps existing callers
        working by rebuilding the LeRobot processors with an action override.
        """
        if mode is not None:
            if mode not in ("absolute", "relative"):
                raise ValueError(f"Unsupported action normalization mode: {mode!r}")
            self._action_norm_mode = mode
        self.configure_processors(
            self._processor_stats,
            action_stats=self._action_stats_from_mean_std(mean, std),
            action_stats_mode=self._action_norm_mode,
        )

    # ------------------------------------------------------------------
    # Persistence (prefer HF-native IO so we get safetensors shards)
    # ------------------------------------------------------------------
    def preprocess_batch(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        """Run the LeRobot SmolVLA preprocessor pipeline."""
        processed = self.preprocessor(dict(batch)) if self.preprocessor is not None else dict(batch)
        if "actions_id_pad" not in processed and "action_is_pad" in processed:
            processed["actions_id_pad"] = processed["action_is_pad"].to(torch.bool)
        elif "actions_id_pad" in processed and isinstance(processed["actions_id_pad"], torch.Tensor):
            processed["actions_id_pad"] = processed["actions_id_pad"].to(torch.bool)
        return processed

    def postprocess_action(self, action: torch.Tensor) -> torch.Tensor:
        """Run the LeRobot SmolVLA postprocessor pipeline."""
        if self.postprocessor is not None:
            action = self.postprocessor(action)
        if self.use_rotation_6d and action.shape[-1] in (10, 11):
            action = action_rotation_6d_to_quat(action)
        return action

    def save_pretrained(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.policy.save_pretrained(save_dir)
        try:
            from lerobot.utils.constants import (
                POLICY_POSTPROCESSOR_DEFAULT_NAME,
                POLICY_PREPROCESSOR_DEFAULT_NAME,
            )
        except ImportError:
            return
        if self.preprocessor is not None:
            self.preprocessor.save_pretrained(
                save_dir,
                config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
            )
        if self.postprocessor is not None:
            self.postprocessor.save_pretrained(
                save_dir,
                config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
            )

    def load_pretrained(self, load_dir: str | Path, strict: bool = True) -> None:
        self.policy = self.policy.__class__.from_pretrained(load_dir, config=self.cfg)
        self.to(self._device)
        if not self._try_load_processors_from_checkpoint(load_dir):
            self.configure_processors()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _move_to_device(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self._device, non_blocking=True)
            else:
                out[k] = v
        return out
