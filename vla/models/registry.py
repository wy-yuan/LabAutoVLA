"""Name-based registry for VLA wrappers.

Used by Hydra configs — set ``model.name: smolvla`` (or ``openvla``, etc.)
and :func:`build_vla` returns the corresponding instance. Keeps the
config files free of Python import paths.
"""

from __future__ import annotations

from typing import Callable, Type

from .base_vla import BaseVLA

_REGISTRY: dict[str, Type[BaseVLA]] = {}


def register_vla(name: str) -> Callable[[Type[BaseVLA]], Type[BaseVLA]]:
    """Class decorator — register a ``BaseVLA`` subclass under ``name``."""

    def _wrap(cls: Type[BaseVLA]) -> Type[BaseVLA]:
        if name in _REGISTRY:
            raise ValueError(f"VLA '{name}' already registered by {_REGISTRY[name]!r}")
        if not issubclass(cls, BaseVLA):
            raise TypeError(f"{cls!r} must subclass BaseVLA")
        _REGISTRY[name] = cls
        cls.name = name
        return cls

    return _wrap


def build_vla(name: str, **kwargs) -> BaseVLA:
    """Instantiate the VLA registered under ``name``."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown VLA '{name}'. Registered: {sorted(_REGISTRY)}. "
            f"Did you import the module that defines it?"
        )
    return _REGISTRY[name](**kwargs)


def list_vlas() -> list[str]:
    return sorted(_REGISTRY)
