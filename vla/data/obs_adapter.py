"""Bridge Matterix observation dicts to the flat dict every VLA expects.

The Matterix env emits a nested dict::

    obs = {
        "camera":        {"overhead_rgb": (N, H, W, 3) float [0,1]},
        "articulations": {"robot__ee_world_pos":  (N, 3),
                          "robot__ee_world_quat": (N, 4),
                          "robot__joint_pos":     (N, 9),
                          "robot__gripper_pos":   (N, 2), ...},
        "rigid_objects": {...},
    }

This module flattens/normalises that into the small set of tensors a VLA
wants — an image dict, a proprioceptive state vector, and a language
prompt — so :mod:`vla.models` never has to touch Matterix-specific keys.

Keeping this conversion in one place means:
    * Both training data conversion AND online rollouts use identical
      feature construction — no train/test skew.
    * Adding a new camera or proprio signal is a one-line config edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch


COMPACT_EE_STATE_DIM = 9
"""[ee_pos(3), ee_quat(4), gripper_pos(2)]."""


@dataclass
class ObsAdapterConfig:
    """Declares which Matterix obs keys feed which VLA inputs.

    Attributes
    ----------
    image_keys:
        Maps ``vla_key -> matterix_path``. ``matterix_path`` is a slash-
        separated path into the nested obs dict. Example:
        ``{"overhead": "camera/overhead_rgb"}``.
    state_keys:
        List of Matterix paths concatenated (last-dim) into
        ``observation.state``. Order matters and must be stable.
    image_size:
        Target (H, W). Images are resized + channel-permuted (HWC->CHW).
    position_history_offsets:
        Positive past-frame offsets appended after the current compact state.
        For example ``[4, 8, 12, 16, 20]`` produces the 24D layout
        ``[state9[t], pos[t-4], ..., pos[t-20]]``. An empty sequence leaves
        the compact state unchanged.
    """

    image_keys: dict[str, str] = field(default_factory=lambda: {"overhead": "camera/overhead_rgb"})
    state_keys: list[str] = field(
        default_factory=lambda: [
            "articulations/robot__ee_world_pos",      # 3
            "articulations/robot__ee_world_quat",     # 4
            "articulations/robot__gripper_pos",       # 2
        ]
    )
    image_size: tuple[int, int] = (224, 224)
    position_history_offsets: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self.image_size = tuple(int(value) for value in self.image_size)
        self.position_history_offsets = tuple(
            int(value) for value in self.position_history_offsets
        )
        validate_position_history_offsets(self.position_history_offsets)


def validate_position_history_offsets(offsets: Sequence[int]) -> tuple[int, ...]:
    """Validate causal history offsets ordered from nearest to farthest."""
    normalized = tuple(int(value) for value in offsets)
    if not normalized:
        return normalized
    if any(value <= 0 for value in normalized):
        raise ValueError("position_history_offsets must contain positive past-frame offsets")
    if any(left >= right for left, right in zip(normalized, normalized[1:])):
        raise ValueError(
            "position_history_offsets must be strictly increasing, "
            "for example [4, 8, 12, 16, 20]"
        )
    return normalized


def model_state_dim(compact_state_dim: int, offsets: Sequence[int]) -> int:
    """Return the flattened model-state width for a compact state schema."""
    normalized = validate_position_history_offsets(offsets)
    if not normalized:
        return int(compact_state_dim)
    if compact_state_dim != COMPACT_EE_STATE_DIM:
        raise ValueError(
            "Position history requires compact 9D state "
            "[ee_pos(3), ee_quat(4), gripper_pos(2)], "
            f"got compact_state_dim={compact_state_dim}"
        )
    return compact_state_dim + 3 * len(normalized)


def augment_state_sequence_with_position_history(
    compact_states: torch.Tensor,
    offsets: Sequence[int],
) -> torch.Tensor:
    """Build one current-first, history-augmented state per episode frame.

    ``compact_states`` must be chronological ``(T, 9)``. Indices before the
    episode start are clamped to frame 0, matching the online reset behavior.
    """
    normalized = validate_position_history_offsets(offsets)
    if not normalized:
        return compact_states
    if compact_states.ndim != 2 or compact_states.shape[-1] != COMPACT_EE_STATE_DIM:
        raise ValueError(
            "Expected chronological compact states with shape (T, 9), "
            f"got {tuple(compact_states.shape)}"
        )
    if compact_states.shape[0] == 0:
        return compact_states.new_empty((0, model_state_dim(COMPACT_EE_STATE_DIM, normalized)))

    frame_ids = torch.arange(compact_states.shape[0], device=compact_states.device)
    past_offsets = torch.as_tensor(normalized, device=compact_states.device)
    history_ids = (frame_ids[:, None] - past_offsets[None, :]).clamp_min(0)
    historical_positions = compact_states[history_ids, :3].flatten(start_dim=1)
    return torch.cat([compact_states, historical_positions], dim=-1)


def compose_state_from_position_buffer(
    position_buffer: torch.Tensor,
    compact_state: torch.Tensor,
    offsets: Sequence[int],
) -> torch.Tensor:
    """Compose current-first online state from a chronological position buffer."""
    normalized = validate_position_history_offsets(offsets)
    if not normalized:
        return compact_state
    if compact_state.ndim != 2 or compact_state.shape[-1] != COMPACT_EE_STATE_DIM:
        raise ValueError(f"Expected compact state (N, 9), got {tuple(compact_state.shape)}")
    required = normalized[-1] + 1
    if position_buffer.ndim != 3 or position_buffer.shape[1:] != (required, 3):
        raise ValueError(
            f"Expected position buffer (N, {required}, 3), got {tuple(position_buffer.shape)}"
        )
    if position_buffer.shape[0] != compact_state.shape[0]:
        raise ValueError("Position buffer and compact state batch sizes do not match")

    # Buffer is oldest -> newest; offsets are emitted nearest -> farthest.
    indices = [required - 1 - offset for offset in normalized]
    historical_positions = position_buffer[:, indices, :].flatten(start_dim=1)
    return torch.cat([compact_state, historical_positions], dim=-1)


def _get_by_path(obs: Mapping[str, Any], path: str) -> torch.Tensor:
    cur: Any = obs
    for part in path.split("/"):
        if part not in cur:
            raise KeyError(f"Obs key '{path}' missing — available at '{part}': {list(cur)}")
        cur = cur[part]
    if not isinstance(cur, torch.Tensor):
        raise TypeError(f"Obs at '{path}' is not a Tensor (got {type(cur).__name__})")
    return cur


def _to_chw(img: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """``(N,H,W,C)`` float [0,1] -> ``(N,C,H',W')`` resized if needed."""
    if img.ndim != 4:
        raise ValueError(f"Expected (N,H,W,C) image, got shape {tuple(img.shape)}")
    img = img.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
    if img.shape[-2:] != target_hw:
        img = torch.nn.functional.interpolate(
            img, size=target_hw, mode="bilinear", align_corners=False
        )
    return img.clamp(0.0, 1.0).float()


def build_vla_inputs(
    obs: Mapping[str, Any],
    cfg: ObsAdapterConfig,
    task: str | Sequence[str],
    num_envs: int | None = None,
) -> dict[str, Any]:
    """Produce the flat dict the VLA wrapper's ``predict_action`` wants.

    Returns
    -------
    dict with:
        ``images[<key>]`` : (N, C, H, W) float32
        ``state``         : (N, D_state)
        ``task``          : list[str] of length N
    """
    images: dict[str, torch.Tensor] = {}
    for vla_key, src_path in cfg.image_keys.items():
        raw = _get_by_path(obs, src_path)
        images[vla_key] = _to_chw(raw, cfg.image_size)

    state_parts = [_get_by_path(obs, p) for p in cfg.state_keys]
    # Each part is (N, d_i) — concat along last dim.
    state = torch.cat([p if p.ndim == 2 else p.unsqueeze(-1) for p in state_parts], dim=-1)

    if num_envs is None:
        num_envs = state.shape[0]
    if isinstance(task, str):
        task_list = [task] * num_envs
    else:
        task_list = list(task)
        assert len(task_list) == num_envs, "task prompts must match batch size"

    return {"images": images, "state": state, "task": task_list}


def state_dim_from_cfg(cfg: ObsAdapterConfig, sample_obs: Mapping[str, Any]) -> int:
    """Compute the history-augmented model state dim from a sample observation."""
    dims = [_get_by_path(sample_obs, p).shape[-1] for p in cfg.state_keys]
    return model_state_dim(sum(dims), cfg.position_history_offsets)
