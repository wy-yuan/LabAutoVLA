"""VLA model wrappers.

All concrete VLAs subclass :class:`vla.models.base_vla.BaseVLA` and are
registered in :mod:`vla.models.registry` so Hydra configs can select them
by name (e.g. ``model.name: smolvla``).

To add a new VLA (OpenVLA, GR00T N1, pi_0, ...):
    1. Create ``vla/models/<name>.py`` with a subclass of ``BaseVLA``.
    2. Register it in ``registry.py`` via ``@register_vla("<name>")``.
    3. Add ``configs/model/<name>.yaml`` with its hyper-parameters.
That's it — no training-code changes required.
"""

from .base_vla import BaseVLA, VLAOutput
from .registry import build_vla, register_vla, list_vlas

# Importing the concrete implementations triggers their registry side-effects.
from . import smolvla  # noqa: F401

__all__ = ["BaseVLA", "VLAOutput", "build_vla", "register_vla", "list_vlas"]
