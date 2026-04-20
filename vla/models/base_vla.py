"""Abstract base class shared by every VLA wrapper in this project.

The contract is deliberately small so that heterogeneous backbones
(SmolVLA, OpenVLA, GR00T N1, pi_0, ...) can all be trained and evaluated
by the SAME training loop and the SAME evaluator. If a new method
genuinely does not fit this contract, prefer extending it here (so all
VLAs benefit) rather than special-casing downstream code.

Conventions
-----------
* Images are ``float32`` tensors in ``[0, 1]`` with layout ``(B, C, H, W)``.
  The adapter in :mod:`vla.data.obs_adapter` is responsible for producing
  them in that form — wrappers should not do ad-hoc resizing here.
* Proprioceptive state is a flat ``(B, D_state)`` float tensor.
* ``task`` is a list of natural-language strings of length ``B``
  (one prompt per sample — training data can have per-episode prompts).
* Actions are ``(B, A)`` float tensors in the env's action space.
  For the Franka IK action space used here, ``A=8``:
  ``[x, y, z, qw, qx, qy, qz, gripper]``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn


@dataclass
class VLAOutput:
    """Standard return type for a forward / predict call.

    Attributes
    ----------
    actions:
        ``(B, chunk_size, A)`` predicted action chunk. For policies that
        predict a single step, ``chunk_size`` is 1.
    loss:
        Optional scalar training loss (populated by ``compute_loss``).
    aux:
        Any model-specific extras — attention maps, latent states, etc.
        Training / eval code must NOT depend on specific keys here.
    """

    actions: torch.Tensor
    loss: torch.Tensor | None = None
    aux: dict[str, Any] | None = None


class BaseVLA(nn.Module, ABC):
    """Abstract VLA policy.

    Subclasses wrap a backbone (often a HuggingFace model) and expose a
    uniform interface for the training loop and the rollout evaluator.
    """

    #: Name used in configs/registry. Subclasses MUST override.
    name: str = "base"

    def __init__(self, action_dim: int, state_dim: int, image_keys: Sequence[str]):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        # Ordered list of camera keys the model expects — keeps multi-cam
        # extensions clean (e.g. adding a wrist cam later).
        self.image_keys = list(image_keys)

    # ---------------------------------------------------------------------
    # Core contract
    # ---------------------------------------------------------------------
    @abstractmethod
    def predict_action(
        self,
        images: Mapping[str, torch.Tensor],
        state: torch.Tensor,
        task: Sequence[str],
    ) -> torch.Tensor:
        """Run inference and return an action tensor of shape ``(B, A)``.

        Must be safe to call under ``torch.inference_mode()``. Wrappers
        that predict action chunks internally should return only the
        first step here (the evaluator handles chunking externally).
        """

    @abstractmethod
    def compute_loss(self, batch: Mapping[str, Any]) -> VLAOutput:
        """Compute the training loss for a dataset batch.

        ``batch`` follows the LeRobot convention:
            * ``batch["observation.images.<key>"]`` — ``(B, T, C, H, W)`` or ``(B, C, H, W)``
            * ``batch["observation.state"]``        — ``(B, T, D)`` or ``(B, D)``
            * ``batch["action"]``                   — ``(B, T_a, A)`` or ``(B, A)``
            * ``batch["task"]``                     — list[str] of length ``B``
        Implementations are free to ignore any of these — e.g. a
        language-free policy can skip ``task``.
        """

    # ---------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------
    def save_pretrained(self, save_dir: str | Path) -> None:
        """Persist weights + minimal metadata to ``save_dir``.

        Default implementation uses ``torch.save`` on the state dict.
        HuggingFace-backed wrappers typically override this to call the
        underlying ``save_pretrained`` and get safetensors shards.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_dir / "model.pt")

    def load_pretrained(self, load_dir: str | Path, strict: bool = True) -> None:
        """Inverse of :meth:`save_pretrained`."""
        load_dir = Path(load_dir)
        sd = torch.load(load_dir / "model.pt", map_location="cpu")
        self.load_state_dict(sd, strict=strict)

    # ---------------------------------------------------------------------
    # Optional hooks (override only when needed)
    # ---------------------------------------------------------------------
    def reset(self) -> None:
        """Called by the evaluator at the start of every episode.

        Useful for models with internal temporal state (e.g. action
        chunking buffers, RNN hidden states). Default: no-op.
        """

    def preprocess_batch(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        """Hook for model-specific batch tweaks (tokenisation, casting).

        Training code calls this once per batch before ``compute_loss``.
        Default: pass-through.
        """
        return batch
