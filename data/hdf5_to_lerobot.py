"""Convert synthetic HDF5 recordings into a LeRobot v3.0 dataset."""

from __future__ import annotations

import argparse
import functools
import logging
import shutil
import tempfile
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


@dataclass
class HDF5DemoRef:
    """Metadata for one HDF5 demo while its file handle is open."""

    source_path: Path
    demo_name: str
    task: str
    task_id: str | None
    success: bool | None
    num_steps: int


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


def _h5_group_to_frame_tensor_dict(group: h5py.Group, frame_index: int) -> dict[str, Any]:
    """Read one frame from a nested HDF5 observation group."""
    out: dict[str, Any] = {}
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            out[key] = _h5_group_to_frame_tensor_dict(value, frame_index)
        else:
            array = np.asarray(value[frame_index : frame_index + 1] if value.ndim >= 1 else value)
            out[key] = torch.from_numpy(array)
    return out


def _h5_group_to_frame_slice_tensor_dict(
    group: h5py.Group,
    start: int,
    stop: int,
) -> dict[str, Any]:
    """Read a contiguous frame slice from a nested HDF5 observation group."""
    out: dict[str, Any] = {}
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            out[key] = _h5_group_to_frame_slice_tensor_dict(value, start, stop)
        else:
            array = np.asarray(value[start:stop] if value.ndim >= 1 else value)
            out[key] = torch.from_numpy(array)
    return out


def _demo_ref_from_group(
    src_path: Path,
    demo_name: str,
    demo_group: h5py.Group,
    default_task: str | None,
) -> HDF5DemoRef:
    """Build lightweight metadata for one HDF5 demo group."""
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
    return HDF5DemoRef(
        source_path=src_path,
        demo_name=demo_name,
        task=task,
        task_id=task_id,
        success=success,
        num_steps=int(demo_group["actions"].shape[0]),
    )


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


def _iter_demo_groups(
    src_paths: Iterable[Path],
    default_task: str | None = None,
) -> Iterator[tuple[HDF5DemoRef, h5py.Group]]:
    """Yield lightweight demo metadata and open HDF5 groups one at a time."""
    for src_path in src_paths:
        with h5py.File(src_path, "r") as handle:
            if "data" not in handle:
                raise ValueError(f"{src_path} has no top-level 'data/' group")

            for demo_name in sorted(handle["data"].keys()):
                demo_group = handle["data"][demo_name]
                yield _demo_ref_from_group(src_path, demo_name, demo_group, default_task), demo_group


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


def _encode_video_worker_with_crf(
    video_key: str,
    episode_index: int,
    root: Path,
    fps: int,
    vcodec: str = "h264",
    encoder_threads: int | None = None,
    *,
    video_crf: int = 18,
) -> Path:
    """Picklable LeRobot video worker that forwards the configured CRF."""
    import lerobot.datasets.lerobot_dataset as lerobot_dataset
    from lerobot.datasets.video_utils import encode_video_frames

    temp_path = Path(tempfile.mkdtemp(dir=root)) / f"{video_key}_{episode_index:03d}.mp4"
    fpath = lerobot_dataset.DEFAULT_IMAGE_PATH.format(
        image_key=video_key, episode_index=episode_index, frame_index=0
    )
    img_dir = (root / fpath).parent
    encode_video_frames(
        img_dir,
        temp_path,
        fps,
        vcodec=vcodec,
        crf=video_crf,
        overwrite=True,
        encoder_threads=encoder_threads,
    )
    shutil.rmtree(img_dir)
    return temp_path


def _install_lerobot_video_encoding_patch(video_crf: int) -> None:
    """Patch LeRobot's internal video worker to use a lower CRF."""
    try:
        import lerobot.datasets.lerobot_dataset as lerobot_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Dataset conversion requires the `lerobot` package "
            "(pip install lerobot). See requirements.txt."
        ) from exc

    lerobot_dataset._encode_video_worker = functools.partial(
        _encode_video_worker_with_crf,
        video_crf=int(video_crf),
    )


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
    video_crf: int,
):
    """Create the destination LeRobot dataset from the first demo."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Dataset conversion requires the `lerobot` package "
            "(pip install lerobot). See requirements.txt."
        ) from exc

    _install_lerobot_video_encoding_patch(video_crf)

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


def _create_dataset_from_hdf5_group(
    first_demo: HDF5DemoRef,
    first_demo_group: h5py.Group,
    dst: Path,
    repo_id: str,
    fps: int,
    adapter_cfg: ObsAdapterConfig,
    use_videos: bool,
    vcodec: str,
    video_crf: int,
):
    """Create a LeRobot dataset by inspecting only the first HDF5 frame."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Dataset conversion requires the `lerobot` package "
            "(pip install lerobot). See requirements.txt."
        ) from exc

    _install_lerobot_video_encoding_patch(video_crf)

    sample_obs = _h5_group_to_frame_tensor_dict(first_demo_group["obs"], 0)
    sample_inputs = build_vla_inputs(sample_obs, adapter_cfg, task=first_demo.task, num_envs=1)
    state_dim = int(sample_inputs["state"].shape[-1])
    action_dim = int(first_demo_group["actions"].shape[-1])
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


def _add_hdf5_demo_to_dataset(
    dataset: Any,
    demo: HDF5DemoRef,
    demo_group: h5py.Group,
    adapter_cfg: ObsAdapterConfig,
    frames_per_chunk: int,
) -> None:
    """Stream one HDF5 demo into an open LeRobot dataset."""
    for start in range(0, demo.num_steps, frames_per_chunk):
        stop = min(start + frames_per_chunk, demo.num_steps)
        chunk_len = stop - start

        obs_chunk = _h5_group_to_frame_slice_tensor_dict(demo_group["obs"], start, stop)
        actions_chunk = torch.from_numpy(np.asarray(demo_group["actions"][start:stop])).float()
        inputs = build_vla_inputs(obs_chunk, adapter_cfg, task=demo.task, num_envs=chunk_len)

        states = inputs["state"].float().cpu().numpy()
        actions = actions_chunk.cpu().numpy()
        images = {
            image_key: (image.clamp(0, 1) * 255.0)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
            for image_key, image in inputs["images"].items()
        }

        for i in range(chunk_len):
            frame = {
                "observation.state": states[i],
                "action": actions[i],
                "task": demo.task,
            }
            for image_key, image_np in images.items():
                frame[f"observation.images.{image_key}"] = image_np[i]

            dataset.add_frame(frame)

    dataset.save_episode()
    log.info(
        "Converted %s:%s (%d steps, task_id=%s, success=%s)",
        demo.source_path.name,
        demo.demo_name,
        demo.num_steps,
        demo.task_id,
        demo.success,
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
    video_crf: int = 18,
    frames_per_chunk: int = 64,
    overwrite: bool = False,
) -> Path:
    """Convert one or more HDF5 recordings into a LeRobot dataset."""
    dst_path = Path(dst)
    frames_per_chunk = int(frames_per_chunk)
    if frames_per_chunk < 1:
        raise ValueError(f"frames_per_chunk must be >= 1, got {frames_per_chunk}")

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

    demo_iter = _iter_demo_groups(src_paths, default_task=task)
    first_item = next(demo_iter, None)
    if first_item is None:
        raise ValueError("No demos were found in the supplied HDF5 files.")

    first_demo, first_demo_group = first_item
    dataset = _create_dataset_from_hdf5_group(
        first_demo=first_demo,
        first_demo_group=first_demo_group,
        dst=dst_path,
        repo_id=repo_id,
        fps=fps,
        adapter_cfg=adapter_cfg,
        use_videos=use_videos,
        vcodec=vcodec,
        video_crf=video_crf,
    )

    log.info("Streaming HDF5 frames in chunks of up to %d frame(s)", frames_per_chunk)
    _add_hdf5_demo_to_dataset(dataset, first_demo, first_demo_group, adapter_cfg, frames_per_chunk)
    for demo, demo_group in demo_iter:
        _add_hdf5_demo_to_dataset(dataset, demo, demo_group, adapter_cfg, frames_per_chunk)

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
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument(
        "--frames-per-chunk",
        type=int,
        default=64,
        help="Maximum number of frames to read and adapt at once during conversion.",
    )
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
        video_crf=args.video_crf,
        frames_per_chunk=args.frames_per_chunk,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    _cli()
