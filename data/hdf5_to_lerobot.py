"""Convert synthetic HDF5 recordings into a LeRobot v3.0 dataset."""

from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import h5py
import numpy as np
import torch

from vla.data.obs_adapter import ObsAdapterConfig, build_vla_inputs

log = logging.getLogger(__name__)

_VCODEC_ALIASES = {
    "libx264": "h264",
    "avc": "h264",
    "x264": "h264",
    "libx265": "hevc",
    "h265": "hevc",
}


@dataclass
class HDF5Demo:
    """One demo loaded from an HDF5 dataset."""

    source_path: Path
    demo_name: str
    observations: dict[str, Any]
    actions: torch.Tensor
    task: str
    task_id: str | None
    success: bool | None


def _decode_attr(value: Any) -> str | None:
    """Decode HDF5 attributes into Python strings."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return str(value)


def _h5_group_to_tensor_dict(group: h5py.Group) -> dict[str, Any]:
    """Convert a nested HDF5 group into a nested tensor dict."""
    out: dict[str, Any] = {}
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            out[key] = _h5_group_to_tensor_dict(value)
        else:
            out[key] = torch.from_numpy(np.asarray(value))
    return out


def _iter_demos(src: str | Path, default_task: str | None = None) -> Iterator[HDF5Demo]:
    """Yield all demos stored in one HDF5 file."""
    src_path = Path(src)
    with h5py.File(src_path, "r") as handle:
        if "data" not in handle:
            raise ValueError(f"{src_path} has no top-level 'data/' group")

        for demo_name in sorted(handle["data"].keys()):
            demo_group = handle["data"][demo_name]
            if "obs" not in demo_group or "actions" not in demo_group:
                raise ValueError(f"{src_path}:{demo_name} is missing required 'obs' or 'actions' data")

            task = _decode_attr(demo_group.attrs.get("task")) or default_task
            if task is None:
                raise ValueError(
                    f"{src_path}:{demo_name} has no stored task prompt. "
                    "Pass a fallback task string to the converter."
                )

            task_id = _decode_attr(demo_group.attrs.get("task_id"))
            success_attr = demo_group.attrs.get("success")
            success = None if success_attr is None else bool(success_attr)

            yield HDF5Demo(
                source_path=src_path,
                demo_name=demo_name,
                observations=_h5_group_to_tensor_dict(demo_group["obs"]),
                actions=torch.from_numpy(np.asarray(demo_group["actions"])),
                task=task,
                task_id=task_id,
                success=success,
            )


def _index_tree(tree: dict[str, Any], t: int) -> dict[str, Any]:
    """Return ``tree[..., t]`` for every leaf tensor (first-dim indexing)."""
    out: dict[str, Any] = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            out[key] = _index_tree(value, t)
        else:
            out[key] = value[t : t + 1] if value.ndim >= 1 else value
    return out


def _normalize_vcodec(vcodec: str) -> str:
    """Map common ffmpeg codec aliases to LeRobot's accepted names."""
    normalized = _VCODEC_ALIASES.get(vcodec, vcodec)
    if normalized != vcodec:
        log.info("Mapped video codec %s -> %s for LeRobot conversion", vcodec, normalized)
    return normalized


def _finalize_dataset(dataset: Any) -> None:
    """Finalize a LeRobot dataset across API versions."""
    if hasattr(dataset, "consolidate"):
        dataset.consolidate()
        return

    close_writer = getattr(dataset, "_close_writer", None)
    if callable(close_writer):
        close_writer()

    stop_image_writer = getattr(dataset, "stop_image_writer", None)
    if callable(stop_image_writer):
        stop_image_writer()


def _create_dataset(
    first_demo: HDF5Demo,
    dst: Path,
    repo_id: str,
    fps: int,
    adapter_cfg: ObsAdapterConfig,
    use_videos: bool,
    vcodec: str,
):
    """Create the destination LeRobot dataset from the first demo."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Dataset conversion requires the `lerobot` package "
            "(pip install lerobot). See requirements.txt."
        ) from exc

    sample_inputs = build_vla_inputs(
        first_demo.observations,
        adapter_cfg,
        task=first_demo.task,
        num_envs=first_demo.actions.shape[0],
    )
    state_dim = int(sample_inputs["state"].shape[-1])
    action_dim = int(first_demo.actions.shape[-1])
    image_shapes = {key: tuple(value.shape[1:]) for key, value in sample_inputs["images"].items()}

    features: dict[str, dict[str, Any]] = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": None},
        "action": {"dtype": "float32", "shape": (action_dim,), "names": None},
    }
    for image_key, shape in image_shapes.items():
        features[f"observation.images.{image_key}"] = {
            "dtype": "video",
            "shape": shape,
            "names": ["channels", "height", "width"],
        }

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=dst,
        features=features,
        use_videos=use_videos,
        vcodec=_normalize_vcodec(vcodec),
    )


def convert(
    src: str | Path | Iterable[str | Path],
    dst: str | Path,
    task: str | None = None,
    fps: int = 30,
    repo_id: str | None = None,
    adapter_cfg: ObsAdapterConfig | None = None,
    use_videos: bool = True,
    vcodec: str = "libx264",
    overwrite: bool = False,
) -> Path:
    """Convert one or more HDF5 recordings into a LeRobot dataset."""
    dst_path = Path(dst)
    src_paths = [Path(p) for p in (src if isinstance(src, (list, tuple, set)) else [src])]
    src_paths = [path for path in src_paths if path.exists()]
    if not src_paths:
        raise FileNotFoundError("No HDF5 source files were found for conversion.")

    if dst_path.exists():
        if overwrite:
            shutil.rmtree(dst_path)
        else:
            raise FileExistsError(
                f"LeRobot dataset already exists: {dst_path}. "
                "Set output.overwrite_lerobot=true to replace it."
            )

    adapter_cfg = adapter_cfg or ObsAdapterConfig(
        image_keys={"overhead": "camera/overhead_rgb"},
        state_keys=[
            "articulations/robot__ee_world_pos",
            "articulations/robot__ee_world_quat",
            "articulations/robot__gripper_pos",
        ],
        image_size=(224, 224),
    )
    repo_id = repo_id or f"local/{dst_path.name}"

    demos: list[HDF5Demo] = []
    for src_path in src_paths:
        demos.extend(list(_iter_demos(src_path, default_task=task)))

    if not demos:
        raise ValueError("No demos were found in the supplied HDF5 files.")

    dataset = _create_dataset(
        first_demo=demos[0],
        dst=dst_path,
        repo_id=repo_id,
        fps=fps,
        adapter_cfg=adapter_cfg,
        use_videos=use_videos,
        vcodec=vcodec,
    )

    for demo in demos:
        num_steps = int(demo.actions.shape[0])
        for t in range(num_steps):
            obs_t = _index_tree(demo.observations, t)
            inputs = build_vla_inputs(obs_t, adapter_cfg, task=demo.task, num_envs=1)

            frame = {
                "observation.state": inputs["state"].squeeze(0).float().cpu().numpy(),
                "action": demo.actions[t].float().cpu().numpy(),
                "task": demo.task,
            }
            for image_key, image in inputs["images"].items():
                image_np = (image.squeeze(0).clamp(0, 1) * 255.0).to(torch.uint8)
                frame[f"observation.images.{image_key}"] = image_np.permute(1, 2, 0).cpu().numpy()

            dataset.add_frame(frame)
        dataset.save_episode()
        log.info(
            "Converted %s:%s (%d steps, task_id=%s, success=%s)",
            demo.source_path.name,
            demo.demo_name,
            num_steps,
            demo.task_id,
            demo.success,
        )

    _finalize_dataset(dataset)
    log.info("Done. LeRobot dataset written to %s", dst_path)
    return dst_path


def _cli() -> None:
    """CLI for converting HDF5 recordings into LeRobot datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", action="append", required=True, help="Input HDF5 file. Repeat for multiple files.")
    parser.add_argument("--dst", required=True, help="Output LeRobot dataset directory.")
    parser.add_argument("--task", default=None, help="Fallback language prompt if HDF5 demos do not store one.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--repo_id", default=None)
    parser.add_argument("--vcodec", default="libx264")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    convert(
        src=args.src,
        dst=args.dst,
        task=args.task,
        fps=args.fps,
        repo_id=args.repo_id,
        vcodec=args.vcodec,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    _cli()
