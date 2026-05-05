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

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .base_vla import BaseVLA, VLAOutput
from .registry import register_vla

log = logging.getLogger(__name__)

_ACTION_NORMALIZATION_FILE = "action_normalization.json"


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
    """Convert learned 8D base-frame action to the Matterix env command."""
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
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = False,
        load_pretrained: bool = True,
        load_vlm_weights: bool = False,
        use_relative_actions: bool = False,
        step0_loss_weight: float = 1.0,
        device: str | torch.device = "cuda",
    ):
        super().__init__(action_dim=action_dim, state_dim=state_dim, image_keys=image_keys)
        self.use_relative_actions = use_relative_actions
        self.step0_loss_weight = step0_loss_weight
        self._action_norm_mean: torch.Tensor | None = None
        self._action_norm_std: torch.Tensor | None = None
        self._action_norm_mode = "relative" if self.use_relative_actions else "absolute"
        # Tracks which step in the current action chunk we are (for inference).
        self._rel_base_state: torch.Tensor | None = None
        self._rel_step_in_chunk: int = 0
        # Match bc_train's temporal observation delta_timestamps:
        # images [-0.2, -0.1, 0.0], state [-0.1, 0.0].
        self._image_obs_steps = 3
        self._state_obs_steps = 2
        self._image_history: dict[str, list[torch.Tensor]] = {}
        self._state_history: list[torch.Tensor] = []

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
        cfg_kwargs["freeze_vision_encoder"] = freeze_vision_encoder
        cfg_kwargs["train_expert_only"] = train_expert_only
        cfg_kwargs["load_vlm_weights"] = load_vlm_weights
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

        self._device = torch.device(device)
        self.to(self._device)
        self._load_action_normalizer_from_checkpoint(pretrained_name_or_path)

    # ------------------------------------------------------------------
    # Contract
    # ------------------------------------------------------------------
    def compute_loss(self, batch: Mapping[str, Any]) -> VLAOutput:
        """Delegate to the underlying policy's forward-with-loss path."""
        batch = self._move_to_device(batch)
        if self.use_relative_actions:
            batch = self._batch_to_relative(batch)
        batch = self._batch_to_normalized_action(batch)
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
            "observation.state": self._stack_temporal_history(
                self._state_history,
                current_state,
                self._state_obs_steps,
            ),
            "task": list(task),
        }
        for k in self.image_keys:
            if k not in images:
                raise KeyError(f"SmolVLA expects image key '{k}' — got {list(images)}")
            image = images[k].to(self._device)
            current_image = image[:, -1].contiguous() if image.ndim == 5 else image
            image_history = self._image_history.setdefault(k, [])
            observation[f"observation.images.{k}"] = self._stack_temporal_history(
                image_history,
                current_image,
                self._image_obs_steps,
            )

        observation = self.preprocess_batch(observation)
        observation = self._move_to_device(observation)

        # ``select_action`` streams one action per call out of the internal
        # chunk buffer; call ``reset()`` between episodes (see BaseVLA.reset).
        action = self.policy.select_action(observation)
        action = self._unnormalize_action(action)

        if self.use_relative_actions and self._rel_base_state is not None:
            action = actions_to_absolute(action, self._rel_base_state)
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
        self._image_history.clear()
        self._state_history.clear()

    @staticmethod
    def _stack_temporal_history(
        history: list[torch.Tensor],
        current: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        """Append current observation and return ``(B, steps, ...)`` history."""
        if (
            not history
            or history[-1].shape != current.shape
            or history[-1].dtype != current.dtype
            or history[-1].device != current.device
        ):
            history[:] = [current.clone() for _ in range(steps)]
        else:
            history.append(current.clone())
            del history[:-steps]
            while len(history) < steps:
                history.insert(0, history[0].clone())
        return torch.stack(history[-steps:], dim=1)

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

    def _batch_to_relative(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Convert the ``action`` field of a training batch to relative actions."""
        processed = dict(batch)
        state = batch.get("observation.state")
        action = batch.get("action")
        if state is None or action is None:
            return processed
        processed["action"] = actions_to_relative(action, state)
        return processed

    def _batch_to_normalized_action(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize the ``action`` field with the installed action stats."""
        processed = dict(batch)
        action = batch.get("action")
        if action is None:
            return processed
        processed["action"] = self._normalize_action(action)
        return processed

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Map env/relative action units into model target units."""
        if self._action_norm_mean is None or self._action_norm_std is None:
            return action
        mean = self._action_norm_mean.to(device=action.device, dtype=action.dtype)
        std = self._action_norm_std.to(device=action.device, dtype=action.dtype)
        return (action - mean) / std.clamp_min(1e-6)

    def _unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Map model output units back into env/relative action units."""
        if self._action_norm_mean is None or self._action_norm_std is None:
            return action
        mean = self._action_norm_mean.to(device=action.device, dtype=action.dtype)
        std = self._action_norm_std.to(device=action.device, dtype=action.dtype)
        return action * std.clamp_min(1e-6) + mean

    def has_action_normalizer_stats(self) -> bool:
        """Return whether explicit action normalization stats are installed."""
        return self._action_norm_mean is not None and self._action_norm_std is not None

    def update_action_normalizer_stats(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        mode: str | None = None,
    ) -> None:
        """Install action stats used by this wrapper and LeRobot processors.

        The wrapper applies these stats explicitly before ``policy.forward``
        and after ``policy.select_action``. We also mirror them into LeRobot's
        config/buffers so checkpoints stay compatible with processor-based
        tooling.
        """
        try:
            from lerobot.utils.constants import ACTION
        except ImportError:
            ACTION = "action"

        dev = self._device
        mean_d = mean.detach().float().to(dev)
        std_d = std.detach().float().clamp(min=1e-6).to(dev)
        self._action_norm_mean = mean_d.clone()
        self._action_norm_std = std_d.clone()
        if mode is not None:
            if mode not in ("absolute", "relative"):
                raise ValueError(f"Unsupported action normalization mode: {mode!r}")
            self._action_norm_mode = mode

        # Update the config record so the stats persist with the checkpoint.
        if hasattr(self.policy, "config") and self.policy.config is not None:
            ds = getattr(self.policy.config, "dataset_stats", None) or {}
            if not isinstance(ds, dict):
                ds = {}
            ds[ACTION] = {
                "mean": mean_d.detach().cpu().tolist(),
                "std": std_d.detach().cpu().tolist(),
            }
            try:
                self.policy.config.dataset_stats = ds
            except Exception:
                pass

        # Try to patch the normalizer module buffers directly.
        # LeRobot stores stats as named buffers; walk candidate modules.
        for attr in ("normalize_targets", "unnormalize_outputs"):
            m = getattr(self.policy, attr, None)
            if m is None:
                continue
            for _, submod in m.named_modules():
                buffers = dict(submod.named_buffers(recurse=False))
                for buf_name, buf in buffers.items():
                    if buf is None:
                        continue
                    if "mean" in buf_name and buf.shape == mean_d.shape:
                        submod._buffers[buf_name] = mean_d.clone()
                    elif "std" in buf_name and buf.shape == std_d.shape:
                        submod._buffers[buf_name] = std_d.clone()

    def _load_action_normalizer_from_checkpoint(self, pretrained_name_or_path: str | Path) -> None:
        """Load explicit action normalization stats saved by ``save_pretrained``."""
        ckpt_dir = Path(pretrained_name_or_path)
        if ckpt_dir.suffix == ".safetensors":
            ckpt_dir = ckpt_dir.parent
        stats_path = ckpt_dir / _ACTION_NORMALIZATION_FILE
        if not stats_path.exists():
            return

        try:
            payload = json.loads(stats_path.read_text(encoding="utf-8"))
            mean = torch.tensor(payload["mean"], dtype=torch.float32)
            std = torch.tensor(payload["std"], dtype=torch.float32)
        except Exception as exc:
            log.warning("Failed to load action normalization stats from %s: %s", stats_path, exc)
            return

        mode = str(payload.get("mode") or self._action_norm_mode)
        if mode in ("absolute", "relative") and mode != self._action_norm_mode:
            log.warning(
                "Checkpoint action normalization mode is %s but config requested %s; "
                "using checkpoint mode.",
                mode,
                self._action_norm_mode,
            )
            self.use_relative_actions = mode == "relative"
            self._action_norm_mode = mode

        self.update_action_normalizer_stats(mean, std, mode=mode)
        log.info("Loaded %s action normalization stats from %s", mode, stats_path)

    # ------------------------------------------------------------------
    # Persistence (prefer HF-native IO so we get safetensors shards)
    # ------------------------------------------------------------------
    def preprocess_batch(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        """Prepare language tokens and compatibility keys expected by LeRobot SmolVLA."""
        try:
            from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
        except ImportError:
            return batch

        processed = dict(batch)

        if OBS_LANGUAGE_TOKENS not in processed and "task" in processed:
            task = processed["task"]
            if isinstance(task, str):
                task_list = [task]
            else:
                task_list = list(task)

            task_list = [text if text.endswith("\n") else f"{text}\n" for text in task_list]
            tokenized = self.policy.model.vlm_with_expert.processor.tokenizer(
                task_list,
                padding=self.cfg.pad_language_to,
                padding_side="right",
                max_length=self.cfg.tokenizer_max_length,
                truncation=True,
                return_tensors="pt",
            )
            processed[OBS_LANGUAGE_TOKENS] = tokenized["input_ids"]
            processed[OBS_LANGUAGE_ATTENTION_MASK] = tokenized["attention_mask"].to(torch.bool)

        if "actions_id_pad" not in processed and "action_is_pad" in processed:
            processed["actions_id_pad"] = processed["action_is_pad"].to(torch.bool)
        elif "actions_id_pad" in processed and isinstance(processed["actions_id_pad"], torch.Tensor):
            processed["actions_id_pad"] = processed["actions_id_pad"].to(torch.bool)

        return processed

    def save_pretrained(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.policy.save_pretrained(save_dir)
        if self._action_norm_mean is not None and self._action_norm_std is not None:
            payload = {
                "mode": self._action_norm_mode,
                "mean": self._action_norm_mean.detach().cpu().tolist(),
                "std": self._action_norm_std.detach().cpu().tolist(),
            }
            (save_dir / _ACTION_NORMALIZATION_FILE).write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )

    def load_pretrained(self, load_dir: str | Path, strict: bool = True) -> None:
        self.policy = self.policy.__class__.from_pretrained(load_dir, config=self.cfg)
        self.to(self._device)
        self._load_action_normalizer_from_checkpoint(load_dir)

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
