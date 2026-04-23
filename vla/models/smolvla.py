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

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .base_vla import BaseVLA, VLAOutput
from .registry import register_vla


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
        device: str | torch.device = "cuda",
    ):
        super().__init__(action_dim=action_dim, state_dim=state_dim, image_keys=image_keys)

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
            self.policy = SmolVLAPolicy.from_pretrained(pretrained_name_or_path, config=self.cfg)
        else:
            self.policy = SmolVLAPolicy(self.cfg)

        self._device = torch.device(device)
        self.to(self._device)

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
        return VLAOutput(actions=None, loss=loss, aux=info)

    @torch.inference_mode()
    def predict_action(
        self,
        images: Mapping[str, torch.Tensor],
        state: torch.Tensor,
        task: Sequence[str],
    ) -> torch.Tensor:
        """Rollout helper — returns a single-step action ``(B, A)``."""
        observation: dict[str, Any] = {
            "observation.state": state.to(self._device),
            "task": list(task),
        }
        for k in self.image_keys:
            if k not in images:
                raise KeyError(f"SmolVLA expects image key '{k}' — got {list(images)}")
            observation[f"observation.images.{k}"] = images[k].to(self._device)

        observation = self.preprocess_batch(observation)
        observation = self._move_to_device(observation)

        # ``select_action`` streams one action per call out of the internal
        # chunk buffer; call ``reset()`` between episodes (see BaseVLA.reset).
        action = self.policy.select_action(observation)
        return action  # (B, action_dim)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear the internal action-chunk queue between episodes."""
        if hasattr(self.policy, "reset"):
            self.policy.reset()

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

    def load_pretrained(self, load_dir: str | Path, strict: bool = True) -> None:
        self.policy = self.policy.__class__.from_pretrained(load_dir, config=self.cfg)
        self.to(self._device)

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
