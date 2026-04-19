"""Incremental HDF5 writer for synthetic dataset collection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

log = logging.getLogger(__name__)


def _to_numpy(value: Any) -> np.ndarray:
    """Convert tensors/arrays/scalars to NumPy for HDF5 storage."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _write_nested(group: h5py.Group, key: str, value: Any) -> None:
    """Recursively write nested dicts to an HDF5 group."""
    if isinstance(value, dict):
        subgroup = group.create_group(key)
        for sub_key, sub_value in value.items():
            _write_nested(subgroup, sub_key, sub_value)
        return

    group.create_dataset(key, data=_to_numpy(value), compression="gzip")


class HDF5DatasetWriter:
    """Write synthetic episodes to an Isaac-Lab-style HDF5 dataset."""

    def __init__(self, output_path: str | Path, overwrite: bool = False):
        self.output_path = Path(output_path)
        if self.output_path.exists():
            if overwrite:
                self.output_path.unlink()
            else:
                raise FileExistsError(
                    f"HDF5 dataset already exists: {self.output_path}. "
                    "Set output.overwrite_hdf5=true to replace it."
                )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.output_path, "w")
        self._data_group = self._file.create_group("data")
        self._data_group.attrs["total"] = 0
        self.episode_count = 0
        self.total_samples = 0

        log.info("Initialized HDF5 dataset writer at %s", self.output_path)

    def add_episode(
        self,
        observations: dict[str, Any],
        actions: torch.Tensor | np.ndarray,
        task: str,
        task_id: str,
        success: bool | None = None,
    ) -> str:
        """Add one episode to the dataset."""
        if not observations:
            raise ValueError("Episode observations are empty.")

        actions_np = _to_numpy(actions).astype(np.float32, copy=False)
        if actions_np.ndim == 0:
            raise ValueError("Episode actions must have at least one dimension.")

        demo_name = f"demo_{self.episode_count}"
        episode_group = self._data_group.create_group(demo_name)
        episode_group.attrs["num_samples"] = int(actions_np.shape[0])
        episode_group.attrs["task"] = task
        episode_group.attrs["task_id"] = task_id
        if success is not None:
            episode_group.attrs["success"] = bool(success)

        _write_nested(episode_group, "obs", observations)
        episode_group.create_dataset("actions", data=actions_np, compression="gzip")

        num_samples = int(actions_np.shape[0])
        self.total_samples += num_samples
        self._data_group.attrs["total"] = self.total_samples
        self.episode_count += 1
        self.flush()
        return demo_name

    def flush(self) -> None:
        """Flush buffered writes to disk."""
        self._file.flush()

    def close(self) -> None:
        """Close the underlying HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def get_stats(self) -> dict[str, Any]:
        """Return basic writer statistics."""
        return {
            "output_path": str(self.output_path),
            "num_episodes": self.episode_count,
            "num_samples": self.total_samples,
        }
