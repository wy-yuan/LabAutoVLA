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
    """Compute the concatenated state dim from a sample observation."""
    dims = [_get_by_path(sample_obs, p).shape[-1] for p in cfg.state_keys]
    return sum(dims)
