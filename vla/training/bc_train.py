"""Behavior Cloning fine-tune for any VLA registered in :mod:`vla.models`.

This is the **main** training entry point. It is called from the Hydra
top-level script ``scripts/train.py`` with ``mode=bc``.

Pipeline
--------
    1. Build the VLA via the registry (``cfg.model.name``).
    2. Load a ``LeRobotDataset`` produced by
       :mod:`data.hdf5_to_lerobot`.
    3. Standard supervised loop: mini-batch -> ``vla.compute_loss`` ->
       Adam/AdamW step -> log -> eval every ``cfg.mode.eval_every`` steps.
    4. On evaluation we boot the Matterix env (only if requested — it's
       expensive) and roll out a few episodes; videos + success rate go
       to the logger.

Design notes
------------
* No assumptions about the model are leaked into this loop. Anything
  SmolVLA-specific (e.g. flow-matching scheduling) lives inside the
  wrapper.
* ``cfg.mode.sim_eval`` is an on/off switch — you can train headlessly
  on a cluster and only run sim evaluation at the end.
* To add a new VLA: register it in ``vla.models`` and point
  ``cfg.model.name`` at it. Nothing else changes.
"""

from __future__ import annotations

import faulthandler
import json
import logging
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from vla.data.obs_adapter import ObsAdapterConfig
log = logging.getLogger(__name__)

_app_launcher = None
simulation_app = None


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------
def _load_task_index_mapping(tasks_path: Path) -> dict[int, str]:
    """Load ``task_index -> task string`` from LeRobot's tasks parquet."""
    import pyarrow.parquet as pq

    table = pq.read_table(tasks_path)
    columns = table.to_pydict()

    if "task_index" not in columns:
        raise ValueError(f"LeRobot tasks metadata is missing 'task_index': {tasks_path}")

    task_name_col = next(
        (name for name, values in columns.items() if name != "task_index" and values),
        None,
    )
    if task_name_col is None:
        raise ValueError(f"Could not infer task name column from {tasks_path}")

    return {
        int(task_index): str(task_name)
        for task_index, task_name in zip(columns["task_index"], columns[task_name_col], strict=True)
    }


def _video_feature_keys(root: Path) -> list[str]:
    """Return video-backed feature keys from ``meta/info.json``."""
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    return [
        feature_name
        for feature_name, feature_spec in info.get("features", {}).items()
        if feature_spec.get("dtype") == "video"
    ]



def _chunk_file_indices(path: Path) -> tuple[int, int]:
    """Parse LeRobot chunk-000/file-000 indices from a dataset path."""
    return int(path.parent.name.split("-")[-1]), int(path.stem.split("-")[-1])


def _video_file_segments(root: Path, video_key: str, fps: int) -> list[dict[str, Any]]:
    """Return cumulative frame ranges for a video key's split MP4 files."""
    from lerobot.datasets.video_utils import get_video_duration_in_s

    video_files = sorted((root / "videos" / video_key).glob("chunk-*/*.mp4"), key=_chunk_file_indices)
    if not video_files:
        raise FileNotFoundError(f"No video files found for LeRobot video key '{video_key}'")

    segments: list[dict[str, Any]] = []
    start_frame = 0
    for video_file in video_files:
        chunk_index, file_index = _chunk_file_indices(video_file)
        duration_s = float(get_video_duration_in_s(video_file))
        frame_count = int(round(duration_s * fps))
        if frame_count <= 0:
            raise ValueError(f"Video file has no frames: {video_file}")

        end_frame = start_frame + frame_count
        segments.append(
            {
                "chunk_index": chunk_index,
                "file_index": file_index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_s": duration_s,
            }
        )
        start_frame = end_frame

    return segments


def _find_video_segment(
    segments: list[dict[str, Any]],
    *,
    episode_index: int,
    video_key: str,
    start_frame: int,
    end_frame: int,
) -> dict[str, Any]:
    for segment in segments:
        if start_frame >= segment["start_frame"] and end_frame <= segment["end_frame"]:
            return segment

    raise ValueError(
        f"Episode {episode_index} frames [{start_frame}, {end_frame}) do not fit in a single "
        f"MP4 segment for video key '{video_key}'."
    )


def _episode_metadata_is_readable(root: Path, episode_files: list[Path], fps: int) -> tuple[bool, str | None]:
    """Return whether all local episode metadata parquet files are readable and usable."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    tables = []
    for path in episode_files:
        try:
            if path.stat().st_size < 8:
                return False, f"{path} is too small to be a parquet file"
            with path.open("rb") as f:
                f.seek(-4, 2)
                if f.read() != b"PAR1":
                    return False, f"{path} is missing the parquet footer magic bytes"
            pq.read_schema(path)
            tables.append(pq.read_table(path))
        except Exception as exc:
            return False, f"{path}: {exc}"

    if not tables:
        return True, None

    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    columns = table.to_pydict()
    video_keys = _video_feature_keys(root)
    for video_key in video_keys:
        required_columns = [
            f"videos/{video_key}/chunk_index",
            f"videos/{video_key}/file_index",
            f"videos/{video_key}/from_timestamp",
            f"videos/{video_key}/to_timestamp",
        ]
        missing_columns = [col for col in required_columns if col not in columns]
        if missing_columns:
            return False, f"episode metadata is missing video columns: {missing_columns}"

        segments = {
            (segment["chunk_index"], segment["file_index"]): segment
            for segment in _video_file_segments(root, video_key, fps)
        }
        tolerance_s = max(1.0 / fps, 1e-3)
        for episode_index, chunk_index, file_index, from_timestamp, to_timestamp in zip(
            columns.get("episode_index", []),
            columns[f"videos/{video_key}/chunk_index"],
            columns[f"videos/{video_key}/file_index"],
            columns[f"videos/{video_key}/from_timestamp"],
            columns[f"videos/{video_key}/to_timestamp"],
            strict=True,
        ):
            segment = segments.get((int(chunk_index), int(file_index)))
            if segment is None:
                return (
                    False,
                    f"episode {episode_index} references missing {video_key} video file "
                    f"chunk={chunk_index} file={file_index}",
                )
            if float(from_timestamp) < -tolerance_s:
                return False, f"episode {episode_index} has negative {video_key} from_timestamp"
            if float(to_timestamp) > segment["duration_s"] + tolerance_s:
                return (
                    False,
                    f"episode {episode_index} {video_key} timestamps exceed referenced video duration",
                )

    return True, None


def _ensure_local_episode_metadata(root: Path, fps: int) -> None:
    """Create or repair ``meta/episodes`` metadata for a local LeRobot dataset."""
    episodes_dir = root / "meta" / "episodes"
    episode_files = sorted(episodes_dir.glob("*/*.parquet"))
    if episode_files:
        is_readable, reason = _episode_metadata_is_readable(root, episode_files, fps)
        if is_readable:
            return
        log.warning("Rebuilding unreadable LeRobot episode metadata: %s", reason)
        shutil.rmtree(episodes_dir)

    data_files = sorted((root / "data").glob("*/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No local parquet data files found under {root / 'data'}")

    tasks_path = root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        raise FileNotFoundError(f"Missing LeRobot tasks metadata: {tasks_path}")

    import pyarrow as pa
    import pyarrow.parquet as pq

    task_index_to_name = _load_task_index_mapping(tasks_path)
    video_keys = _video_feature_keys(root)
    video_segments = {video_key: _video_file_segments(root, video_key, fps) for video_key in video_keys}
    episode_records: dict[int, dict[str, Any]] = {}

    for data_file in data_files:
        table = pq.read_table(data_file, columns=["episode_index", "index", "task_index"])
        columns = table.to_pydict()
        chunk_index = int(data_file.parent.name.split("-")[-1])
        file_index = int(data_file.stem.split("-")[-1])

        for episode_index, frame_index, task_index in zip(
            columns["episode_index"],
            columns["index"],
            columns["task_index"],
            strict=True,
        ):
            ep_idx = int(episode_index)
            abs_idx = int(frame_index)
            task_idx = int(task_index)
            record = episode_records.setdefault(
                ep_idx,
                {
                    "episode_index": ep_idx,
                    "tasks": set(),
                    "length": 0,
                    "dataset_from_index": abs_idx,
                    "dataset_to_index": abs_idx + 1,
                    "data/chunk_index": chunk_index,
                    "data/file_index": file_index,
                },
            )
            record["tasks"].add(task_index_to_name.get(task_idx, str(task_idx)))
            record["length"] += 1
            record["dataset_from_index"] = min(record["dataset_from_index"], abs_idx)
            record["dataset_to_index"] = max(record["dataset_to_index"], abs_idx + 1)

    if not episode_records:
        raise ValueError(f"No episode rows found in local dataset under {root}")

    rows: list[dict[str, Any]] = []
    for ep_idx in sorted(episode_records):
        record = episode_records[ep_idx]
        row = {
            "episode_index": int(record["episode_index"]),
            "tasks": sorted(record["tasks"]),
            "length": int(record["length"]),
            "dataset_from_index": int(record["dataset_from_index"]),
            "dataset_to_index": int(record["dataset_to_index"]),
            "data/chunk_index": int(record["data/chunk_index"]),
            "data/file_index": int(record["data/file_index"]),
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }

        for video_key in video_keys:
            segment = _find_video_segment(
                video_segments[video_key],
                episode_index=row["episode_index"],
                video_key=video_key,
                start_frame=row["dataset_from_index"],
                end_frame=row["dataset_to_index"],
            )
            row[f"videos/{video_key}/chunk_index"] = segment["chunk_index"]
            row[f"videos/{video_key}/file_index"] = segment["file_index"]
            row[f"videos/{video_key}/from_timestamp"] = (
                row["dataset_from_index"] - segment["start_frame"]
            ) / fps
            row[f"videos/{video_key}/to_timestamp"] = (row["dataset_to_index"] - segment["start_frame"]) / fps

        rows.append(row)

    out_path = episodes_dir / "chunk-000" / "file-000.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out_path, compression="snappy")

    log.info("Rebuilt local LeRobot episode metadata at %s", episodes_dir)


def _build_dataset(cfg: DictConfig):
    """Load a LeRobotDataset. If it doesn't exist, convert from HDF5."""
    log.info("Preparing LeRobot dataset from root=%s", cfg.dataset.root)
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install lerobot") from exc

    root = Path(cfg.dataset.root)
    if not root.exists():
        log.info("Dataset %s missing — running conversion from %s", root, cfg.dataset.src_hdf5)
        from data.hdf5_to_lerobot import convert
        convert(
            src=cfg.dataset.src_hdf5,
            dst=root,
            task=cfg.task.prompt,
            fps=cfg.dataset.fps,
            repo_id=cfg.dataset.repo_id,
        )

    # Some locally converted datasets may be missing meta/episodes, which
    # makes LeRobot fall back to Hub metadata download. Repair locally first.
    _ensure_local_episode_metadata(root, fps=int(cfg.dataset.fps))

    # Only action needs temporal expansion here. SmolVLA consumes the current
    # image/state; adding image/state histories just creates redundant tensors.
    delta_timestamps = None
    if cfg.dataset.get("action_chunk_size", None):
        k = int(cfg.dataset.action_chunk_size)
        delta_timestamps = {"action": [i / cfg.dataset.fps for i in range(k)]}

    dataset = LeRobotDataset(
        repo_id=cfg.dataset.repo_id,
        root=root,
        delta_timestamps=delta_timestamps,
    )
    return dataset


def _valid_action_rows(batch: dict[str, Any], action: torch.Tensor) -> torch.Tensor | None:
    """Return a flattened valid-step mask for chunked actions, if available."""
    if action.ndim != 3:
        return None

    B, T = action.shape[:2]
    valid = torch.ones(B, T, dtype=torch.bool)
    pad = batch.get("actions_id_pad", batch.get("action_is_pad"))
    if pad is None:
        return valid.flatten()

    pad = torch.as_tensor(pad, dtype=torch.bool)
    if pad.ndim == 0:
        pad = pad.expand(B, T)
    elif pad.ndim == 1:
        pad = pad[:, None].expand(B, T)
    else:
        pad = pad[:, :T]
        if pad.shape[1] < T:
            fill = torch.zeros(B, T - pad.shape[1], dtype=torch.bool)
            pad = torch.cat([pad, fill], dim=1)

    valid = ~pad
    return valid.flatten()


def _compute_action_stats_from_lerobot_parquet(
    dataset,
    *,
    use_relative_actions: bool,
    use_rotation_6d: bool,
    action_chunk_size: int | None,
) -> dict[str, torch.Tensor] | None:
    """Fast path for local LeRobot datasets that avoids decoding videos."""
    root = getattr(dataset, "root", None)
    if root is None:
        return None

    root = Path(root)
    data_files = sorted((root / "data").glob("*/*.parquet"))
    if not data_files:
        return None

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from vla.models.smolvla import action_quat_to_rotation_6d, actions_to_relative
    except ImportError:
        return None

    columns = ["observation.state", "action", "episode_index", "index"]
    tables = [pq.read_table(path, columns=columns) for path in data_files]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    data = table.to_pydict()

    states = torch.tensor(data["observation.state"], dtype=torch.float32)
    actions = torch.tensor(data["action"], dtype=torch.float32)
    episode_indices = torch.tensor(data["episode_index"], dtype=torch.long)
    row_indices = torch.tensor(data["index"], dtype=torch.long)
    chunk_size = int(action_chunk_size or 1)

    targets: list[torch.Tensor] = []
    for episode_idx in torch.unique(episode_indices, sorted=True):
        ep_mask = episode_indices == episode_idx
        order = torch.argsort(row_indices[ep_mask])
        ep_states = states[ep_mask][order]
        ep_actions = actions[ep_mask][order]
        n_steps = int(ep_actions.shape[0])

        for horizon in range(chunk_size):
            n_valid = n_steps - horizon
            if n_valid <= 0:
                break
            horizon_actions = ep_actions[horizon : horizon + n_valid]
            if use_relative_actions:
                horizon_states = ep_states[:n_valid]
                horizon_actions = actions_to_relative(horizon_actions, horizon_states)
            if use_rotation_6d:
                horizon_actions = action_quat_to_rotation_6d(horizon_actions)
            targets.append(horizon_actions)

    if not targets:
        return None

    all_targets = torch.cat(targets, dim=0)
    std, mean = torch.std_mean(all_targets, dim=0)
    return {
        "mean": mean,
        "std": std.clamp(min=1e-6),
        "min": all_targets.min(dim=0).values,
        "max": all_targets.max(dim=0).values,
    }


def _compute_action_normalization_stats(
    dataset,
    *,
    use_relative_actions: bool,
    use_rotation_6d: bool = False,
    action_chunk_size: int | None = None,
    batch_size: int = 256,
    num_workers: int = 0,
) -> dict[str, torch.Tensor]:
    """Compute mean/std/min/max for model action targets over the dataset.

    For absolute-action training, this is computed from raw dataset action
    chunks. For relative-action training, each action chunk is first expressed
    relative to the current observation state, matching SmolVLA.compute_loss.

    Returns a dict with keys 'mean', 'std', 'min', 'max', each (action_dim,).
    """
    parquet_stats = _compute_action_stats_from_lerobot_parquet(
        dataset,
        use_relative_actions=use_relative_actions,
        use_rotation_6d=use_rotation_6d,
        action_chunk_size=action_chunk_size,
    )
    if parquet_stats is not None:
        return parquet_stats

    from torch.utils.data import DataLoader
    from vla.models.smolvla import action_quat_to_rotation_6d, actions_to_relative

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    chunks: list[torch.Tensor] = []
    for batch in loader:
        action = batch["action"]             # (B, T, action_dim) or (B, action_dim)
        target = action.float()
        if use_relative_actions:
            state = batch["observation.state"]   # (B, state_dim)
            target = actions_to_relative(target, state.float())
        if use_rotation_6d:
            target = action_quat_to_rotation_6d(target)

        if target.ndim == 3:
            valid = _valid_action_rows(batch, target)
            target = target.flatten(0, 1)          # (B*T, action_dim)
            if valid is not None:
                target = target[valid.to(target.device)]
        chunks.append(target.cpu())

    all_targets = torch.cat(chunks, dim=0)       # (N, action_dim)
    std, mean = torch.std_mean(all_targets, dim=0)
    return {
        "mean": mean,
        "std": std.clamp(min=1e-6),
        "min": all_targets.min(dim=0).values,
        "max": all_targets.max(dim=0).values,
    }


def _compute_relative_action_stats(
    dataset,
    action_chunk_size: int | None = None,
    batch_size: int = 256,
    num_workers: int = 0,
) -> dict[str, torch.Tensor]:
    """Backward-compatible wrapper for relative-action stats."""
    return _compute_action_normalization_stats(
        dataset,
        use_relative_actions=True,
        use_rotation_6d=False,
        action_chunk_size=action_chunk_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def _build_optimizer(model: torch.nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.mode.optim.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.mode.optim.lr, betas=tuple(cfg.mode.optim.betas))


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    scheduler_name = cfg.mode.get("scheduler", None)
    if scheduler_name is None:
        return None

    scheduler_name = str(scheduler_name).lower()
    warmup_steps = int(cfg.mode.get("scheduler_warmup_steps", 0))
    total_steps = max(1, int(total_steps))

    if scheduler_name != "cosine":
        raise ValueError(f"Unsupported BC scheduler '{scheduler_name}' (expected 'cosine' or null)")

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, float(current_step - warmup_steps) / float(decay_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _resolve_total_train_steps(cfg: DictConfig, train_loader_len: int) -> int:
    """Return the optimizer-step training horizon.

    ``mode.steps`` is the BC convention. ``mode.epochs`` is accepted only as
    a legacy fallback so older command lines fail gently.
    """
    steps = cfg.mode.get("steps", None)
    if steps is not None:
        total_steps = int(steps)
    else:
        epochs = cfg.mode.get("epochs", None)
        if epochs is None:
            raise ValueError("BC training requires mode.steps")
        total_steps = int(epochs) * int(train_loader_len)
        log.warning(
            "mode.epochs is deprecated for BC; use mode.steps=%d instead",
            total_steps,
        )

    if total_steps <= 0:
        raise ValueError(f"mode.steps must be positive, got {total_steps}")
    return total_steps


def _episode_indices_from_lerobot_dataset(dataset) -> dict[int, list[int]]:
    """Return dataset sample indices grouped by episode index."""
    root = getattr(dataset, "root", None)
    if root is not None:
        root = Path(root)
        data_files = sorted((root / "data").glob("*/*.parquet"))
        if data_files:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq

                tables = [pq.read_table(path, columns=["episode_index", "index"]) for path in data_files]
                table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
                columns = table.to_pydict()
                records = sorted(
                    (int(row_index), int(episode_index))
                    for row_index, episode_index in zip(columns["index"], columns["episode_index"])
                )
                row_indices = [row_index for row_index, _ in records]
                if (
                    len(row_indices) == len(dataset)
                    and len(set(row_indices)) == len(row_indices)
                    and all(0 <= row_index < len(dataset) for row_index in row_indices)
                ):
                    indexed_records = records
                else:
                    indexed_records = [
                        (sample_index, episode_index)
                        for sample_index, (_, episode_index) in enumerate(records)
                    ]

                by_episode: dict[int, list[int]] = {}
                for sample_index, episode_index in indexed_records:
                    by_episode.setdefault(episode_index, []).append(sample_index)
                if by_episode:
                    return by_episode
            except Exception as exc:
                log.warning("Falling back to dataset scan for episode split: %s", exc)

    by_episode: dict[int, list[int]] = {}
    for sample_index in range(len(dataset)):
        sample = dataset[sample_index]
        episode_index = int(torch.as_tensor(sample["episode_index"]).item())
        by_episode.setdefault(episode_index, []).append(sample_index)
    return by_episode


def _split_train_val_dataset(dataset, val_fraction: float, seed: int):
    """Split one dataset into train/validation subsets at episode granularity."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"mode.validation_fraction must be in (0, 1), got {val_fraction}")

    by_episode = _episode_indices_from_lerobot_dataset(dataset)
    episode_ids = sorted(by_episode)
    if len(episode_ids) < 2:
        raise ValueError("Need at least 2 episodes to create an episode-level train/validation split")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(episode_ids), generator=generator).tolist()
    val_episode_count = max(1, int(round(len(episode_ids) * val_fraction)))
    val_episode_count = min(val_episode_count, len(episode_ids) - 1)
    val_episodes = {episode_ids[i] for i in perm[:val_episode_count]}
    train_episodes = set(episode_ids) - val_episodes

    train_indices = sorted(
        sample_index
        for episode_index in train_episodes
        for sample_index in by_episode[episode_index]
    )
    val_indices = sorted(
        sample_index
        for episode_index in val_episodes
        for sample_index in by_episode[episode_index]
    )
    if not train_indices or not val_indices:
        raise ValueError(
            "Episode-level split produced an empty subset: "
            f"episodes={len(episode_ids)} validation_fraction={val_fraction}"
        )

    log.info(
        "Episode-level split: train_episodes=%d val_episodes=%d train_samples=%d val_samples=%d",
        len(train_episodes),
        len(val_episodes),
        len(train_indices),
        len(val_indices),
    )
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def _batch_size(batch: dict[str, Any]) -> int:
    for value in batch.values():
        if isinstance(value, torch.Tensor):
            return int(value.shape[0])
    return 1


@torch.no_grad()
def _validate(vla: torch.nn.Module, loader: DataLoader) -> float:
    """Return mean validation model loss over a held-out loader."""
    vla.eval()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        batch = vla.preprocess_batch(batch)
        out = vla.compute_loss(batch)
        if out.loss is None:
            raise RuntimeError("compute_loss returned no validation loss")
        n = _batch_size(batch)
        total_loss += float(out.loss.detach()) * n
        total_samples += n
    if total_samples == 0:
        raise RuntimeError("Validation loader produced no batches")
    return total_loss / total_samples


# ---------------------------------------------------------------------
# Evaluation harness (Matterix-in-the-loop)
# ---------------------------------------------------------------------
def _launch_sim_app():
    """Launch Isaac Sim on demand for BC evaluation."""
    global _app_launcher, simulation_app
    if simulation_app is not None:
        return simulation_app

    print("[bc_train] Launching Isaac Sim for evaluator...", flush=True)
    from isaaclab.app import AppLauncher

    _app_launcher = AppLauncher(headless=True, enable_cameras=True, livestream=2)
    simulation_app = _app_launcher.app
    print("[bc_train] Isaac Sim launcher returned control.", flush=True)
    return simulation_app


def _close_sim_app() -> None:
    """Close Isaac Sim if BC evaluation launched it."""
    global _app_launcher, simulation_app
    if simulation_app is not None:
        simulation_app.close()
        simulation_app = None
        _app_launcher = None


class _LazyEvaluator:
    """Build the simulator-backed evaluator only when first needed."""

    def __init__(self, cfg: DictConfig, adapter_cfg: ObsAdapterConfig):
        self.cfg = cfg
        self.adapter_cfg = adapter_cfg
        self.env = None
        self.evaluator = None

    def _ensure_built(self) -> None:
        if self.evaluator is not None:
            return

        _launch_sim_app()

        # Heavy imports live inside the factory so BC-only training on
        # machines without Isaac Lab still works (just set sim_eval=false).
        import gymnasium as gym
        import matterix_tasks  # noqa: F401 - registers envs
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from vla.envs.vla_env_wrapper import VLAEnvWrapper
        from vla.training.evaluator import RolloutEvaluator

        env_cfg = parse_env_cfg(self.cfg.task.id, device=self.cfg.mode.device, num_envs=1)
        # if not self.cfg.mode.eval.get("use_async_envs", True):
        #     env_cfg.use_async_envs = False
        spec = gym.spec(self.cfg.task.id)
        log.info(
            "Building evaluator env id=%s entry_point=%s kwargs=%s device=%s num_envs=%s",
            self.cfg.task.id,
            spec.entry_point,
            sorted(spec.kwargs.keys()),
            self.cfg.mode.device,
            getattr(env_cfg.scene, "num_envs", None),
        )

        if not faulthandler.is_enabled():
            faulthandler.enable(file=sys.stderr)
        faulthandler.dump_traceback_later(
            30,
            repeat=True,
            file=sys.stderr,
            exit=False,
        )
        try:
            log.info("Starting evaluator env construction with gym.make")
            env = gym.make(self.cfg.task.id, cfg=env_cfg).unwrapped
            log.info("Evaluator env constructed type=%s", type(env).__name__)
        except Exception:
            log.exception(
                "Failed while constructing evaluator env id=%s entry_point=%s device=%s",
                self.cfg.task.id,
                spec.entry_point,
                self.cfg.mode.device,
            )
            raise
        finally:
            faulthandler.cancel_dump_traceback_later()

        self.env = VLAEnvWrapper(
            env,
            adapter_cfg=self.adapter_cfg,
            task_prompt=self.cfg.task.prompt,
        )
        print(f"[bc_train] VLAEnvWrapper created: {type(self.env).__name__}", flush=True)

        log.info("Building evaluator: VLAEnvWrapper created")
        self.evaluator = RolloutEvaluator(
            env=self.env,
            n_episodes=self.cfg.mode.eval.n_episodes,
            max_steps=self.cfg.mode.eval.max_steps,
            video_dir=Path(self.cfg.output_dir) / "eval_videos",
            video_fps=self.cfg.dataset.fps,
        )
        print("[bc_train] RolloutEvaluator created", flush=True)

    def run(self, policy, step):
        self._ensure_built()
        return self.evaluator.run(policy, step=step)

    def close(self) -> None:
        try:
            if self.env is not None:
                self.env.close()
                self.env = None
        finally:
            self.evaluator = None
            _close_sim_app()


def _build_evaluator(cfg: DictConfig, adapter_cfg: ObsAdapterConfig):
    """Return a closeable lazy evaluator, or ``None`` when disabled."""
    if not cfg.mode.get("sim_eval", True):
        return None
    return _LazyEvaluator(cfg, adapter_cfg)

    """Spin up the Matterix env + wrapper + evaluator, lazily.

    Returned callable is ``(policy, step) -> metrics``. Returns ``None``
    if sim evaluation is disabled.
    """
    if not cfg.mode.get("sim_eval", True):
        return None

    # Heavy imports live inside the factory so BC-only training on
    # machines without Isaac Lab still works (just set sim_eval=false).
    import gymnasium as gym
    import matterix_tasks  # noqa: F401  — registers envs
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from vla.envs.vla_env_wrapper import VLAEnvWrapper

    env_cfg = parse_env_cfg(cfg.task.id, device=cfg.mode.device, num_envs=1)
    spec = gym.spec(cfg.task.id)
    log.info(
        "Building evaluator env id=%s entry_point=%s kwargs=%s device=%s num_envs=%s",
        cfg.task.id,
        spec.entry_point,
        sorted(spec.kwargs.keys()),
        cfg.mode.device,
        getattr(env_cfg.scene, "num_envs", None),
    )

    if not faulthandler.is_enabled():
        faulthandler.enable(file=sys.stderr)
    faulthandler.dump_traceback_later(
        30,
        repeat=True,
        file=sys.stderr,
        exit=False,
    )
    try:
        log.info("Starting evaluator env construction with gym.make")
        env = gym.make(cfg.task.id, cfg=env_cfg).unwrapped
        log.info("Evaluator env constructed type=%s", type(env).__name__)
    except Exception:
        log.exception(
            "Failed while constructing evaluator env id=%s entry_point=%s device=%s",
            cfg.task.id,
            spec.entry_point,
            cfg.mode.device,
        )
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()

    log.info("Evaluator env constructed successfully")

    env = VLAEnvWrapper(env, adapter_cfg=adapter_cfg, task_prompt=cfg.task.prompt)
    print(f"[bc_train] VLAEnvWrapper created: {type(env).__name__}", flush=True)

    log.info("Building evaluator: VLAEnvWrapper created !!!!")
    from vla.training.evaluator import RolloutEvaluator

    evaluator = RolloutEvaluator(
        env=env,
        n_episodes=cfg.mode.eval.n_episodes,
        max_steps=cfg.mode.eval.max_steps,
        video_dir=Path(cfg.output_dir) / "eval_videos",
        video_fps=cfg.dataset.fps,
    )
    print("[bc_train] RolloutEvaluator created", flush=True)

    def _run(policy, step):
        return evaluator.run(policy, step=step)

    return _run


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def run_bc(cfg: DictConfig) -> None:
    log.info("BC config:\n%s", OmegaConf.to_yaml(cfg))
    device = torch.device(cfg.mode.device)

    # Keep simulator startup lazy so Windows DataLoader workers are spawned
    # before Isaac Sim exists in the parent process.
    adapter_cfg = ObsAdapterConfig(**OmegaConf.to_container(cfg.task.adapter, resolve=True))
    eval_runner = _build_evaluator(cfg, adapter_cfg)

    # -- Dataset -------------------------------------------------------
    log.info("Building training dataset")
    print("[bc_train] Building training dataset", flush=True)
    dataset = _build_dataset(cfg)
    train_dataset, val_dataset = _split_train_val_dataset(
        dataset,
        val_fraction=float(cfg.mode.get("validation_fraction", 0.1)),
        seed=int(cfg.mode.get("split_seed", 42)),
    )
    log.info(
        "Dataset ready: total=%d train=%d val=%d",
        len(dataset),
        len(train_dataset),
        len(val_dataset),
    )
    print(
        f"[bc_train] Dataset split: train={len(train_dataset)} val={len(val_dataset)}",
        flush=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.mode.batch_size,
        shuffle=True,
        num_workers=cfg.mode.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.mode.batch_size,
        shuffle=False,
        num_workers=cfg.mode.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    if len(train_loader) == 0:
        raise ValueError(
            "Training loader produced no batches; reduce mode.batch_size or add more data"
        )

    # -- Model ---------------------------------------------------------
    # Infer proprio state dim from a sample batch so we don't hardcode it.
    log.info("Loading sample to infer model dimensions")
    sample = dataset[0]
    for k in sample.keys():
        if "image" in k.lower():
            print("!@!", k, sample[k].shape if hasattr(sample[k], 'shape') else type(sample[k]))
    state_dim = int(sample["observation.state"].shape[-1])
    raw_action_dim = int(sample["action"].shape[-1])
    if raw_action_dim != 8:
        raise ValueError(
            "Expected 8D base-frame actions "
            "[ee_pos_base(3), ee_quat_base(4), gripper_open(1)], "
            f"but dataset action_dim={raw_action_dim}. Regenerate/reconvert the dataset "
            "with data.generate_dataset after the base-frame action patch."
        )
    if state_dim != 9:
        raise ValueError(
            "Expected compact 9D observation.state "
            "[ee_pos(3), ee_quat(4), gripper_pos(2)], "
            f"but dataset state_dim={state_dim}. Regenerate/reconvert the dataset "
            "with the updated task adapter."
        )
    image_keys = [
        k.replace("observation.images.", "")
        for k in sample.keys()
        if k.startswith("observation.images.")
    ]

    model_kwargs = OmegaConf.to_container(cfg.model.kwargs, resolve=True)
    use_rotation_6d_cfg = bool(model_kwargs.get("use_rotation_6d", True))
    if cfg.model.name == "smolvla":
        from vla.models.smolvla import learned_action_dim_from_raw

        action_dim = learned_action_dim_from_raw(
            raw_action_dim,
            use_rotation_6d=use_rotation_6d_cfg,
        )
    else:
        action_dim = raw_action_dim

    log.info(
        "Building VLA model name=%s state_dim=%d raw_action_dim=%d model_action_dim=%d image_keys=%s",
        cfg.model.name,
        state_dim,
        raw_action_dim,
        action_dim,
        image_keys,
    )
    from vla.models import build_vla

    vla = build_vla(
        cfg.model.name,
        action_dim=action_dim,
        state_dim=state_dim,
        image_keys=cfg.task.adapter.image_keys,
        **model_kwargs,
    )
    vla.to(device)
    log.info("Model ready on device=%s", device)

    requested_relative = bool(model_kwargs.get("use_relative_actions", False))
    use_relative = bool(getattr(vla, "use_relative_actions", requested_relative))
    use_rotation_6d = bool(getattr(vla, "use_rotation_6d", use_rotation_6d_cfg))
    if use_relative or use_rotation_6d:
        mode = "relative" if use_relative else "absolute"
        log.info(
            "Computing %s action stats from dataset (use_rotation_6d=%s)",
            mode,
            use_rotation_6d,
        )
        print(f"[bc_train] Computing {mode} action stats...", flush=True)
        action_stats = _compute_action_normalization_stats(
            dataset,
            use_relative_actions=use_relative,
            use_rotation_6d=use_rotation_6d,
            action_chunk_size=int(cfg.dataset.get("action_chunk_size", 1)),
            batch_size=int(cfg.mode.batch_size) * 4,
            num_workers=int(cfg.mode.num_workers),
        )
        log.info(
            "%s action stats: mean=%s std=%s",
            mode.title(),
            action_stats["mean"].tolist(),
            action_stats["std"].tolist(),
        )
        vla.configure_processors(
            dataset.meta.stats,
            action_stats=action_stats,
            action_stats_mode=mode,
        )
        print(f"[bc_train] LeRobot processors configured with {mode} action stats.", flush=True)
    else:
        log.info("use_relative_actions=False; using LeRobot dataset.meta.stats for processors")
        vla.configure_processors(dataset.meta.stats, action_stats_mode="absolute")
        print("[bc_train] LeRobot processors configured from dataset.meta.stats.", flush=True)

    # -- Optim + Logger ------------------------------------------------
    optim = _build_optimizer(vla, cfg)
    total_train_steps = _resolve_total_train_steps(cfg, len(train_loader))
    scheduler = _build_scheduler(optim, cfg, total_train_steps)
    if scheduler is not None:
        log.info(
            "Using %s scheduler with warmup_steps=%d total_steps=%d",
            cfg.mode.scheduler,
            int(cfg.mode.get("scheduler_warmup_steps", 0)),
            total_train_steps,
        )
    from vla.utils.logging import Logger

    logger = Logger(
        log_dir=Path(cfg.output_dir) / "tb",
        use_wandb=cfg.mode.get("use_wandb", False),
        wandb_project=cfg.mode.get("wandb_project", "labauto-vla"),
        wandb_run_name=cfg.mode.get("run_name", None),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    if logger.wandb_url:
        log.info("Weights & Biases run: %s", logger.wandb_url)
        print(f"[bc_train] W&B run: {logger.wandb_url}", flush=True)
    else:
        tb_dir = Path(cfg.output_dir) / "tb"
        log.info("TensorBoard logs: %s", tb_dir)
        print(f"[bc_train] TensorBoard logs: {tb_dir}", flush=True)

    # -- Loop ----------------------------------------------------------
    try:
        ckpt_every = int(cfg.mode.get("ckpt_every", 0) or 0)
        val_every = int(cfg.mode.get("val_every", 0) or 0)
        eval_every = int(cfg.mode.get("eval_every", 0) or 0)
        global_step = 0
        data_epoch = 0
        last_loss = float("nan")
        last_val_step = 0
        last_ckpt_step = 0

        def _save_checkpoint(step: int) -> None:
            nonlocal last_ckpt_step
            ckpt_dir = Path(cfg.output_dir) / f"ckpt_step{step:06d}"
            vla.save_pretrained(ckpt_dir)
            last_ckpt_step = step
            log.info("Saved checkpoint to %s", ckpt_dir)
            print(f"[bc_train] Saved checkpoint at step {step} !!!", flush=True)

        def _run_validation(step: int, *, final: bool = False) -> float:
            nonlocal last_val_step
            val_loss = _validate(vla, val_loader)
            logger.scalar("val/loss", val_loss, step)
            last_val_step = step
            vla.train()
            if final:
                log.info("final val@step%d: %.4f", step, val_loss)
            else:
                log.info("val@step%d: %.4f", step, val_loss)
            return val_loss

        def _run_eval(step: int) -> None:
            if eval_runner is None:
                return
            print(f"[bc_train] Running evaluation at step {step} !!!", flush=True)
            try:
                metrics = eval_runner.run(vla, step=step)
            finally:
                vla.train()
            print("[bc_train] Evaluation complete !!!", flush=True)
            for k, v in metrics.items():
                logger.scalar(f"eval/{k}", v, step)
            log.info("eval@step%d: %s", step, metrics)
            print("[bc_train] eval@step%d: %s" % (step, metrics), flush=True)

        while global_step < total_train_steps:
            data_epoch += 1
            vla.train()
            log.info(
                "Starting training data pass %d at step %d/%d",
                data_epoch,
                global_step,
                total_train_steps,
            )
            for batch in train_loader:
                if global_step >= total_train_steps:
                    break
                batch = vla.preprocess_batch(batch)
                out = vla.compute_loss(batch)
                loss = out.loss
                if loss is None:
                    raise RuntimeError(f"{cfg.model.name}.compute_loss returned no loss")
                optim.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.mode.optim.grad_clip:
                    torch.nn.utils.clip_grad_norm_(vla.parameters(), cfg.mode.optim.grad_clip)
                optim.step()
                if scheduler is not None:
                    scheduler.step()

                last_loss = float(loss.detach())
                global_step += 1
                logger.scalar("train/loss", last_loss, global_step)
                logger.scalar("train/lr", float(optim.param_groups[0]["lr"]), global_step)
                if out.aux:
                    for key, value in out.aux.items():
                        if isinstance(value, torch.Tensor) and value.numel() == 1:
                            logger.scalar(f"train/aux/{key}", float(value.detach()), global_step)
                        elif isinstance(value, (int, float)):
                            logger.scalar(f"train/aux/{key}", float(value), global_step)

                if ckpt_every > 0 and global_step % ckpt_every == 0:
                    _save_checkpoint(global_step)

                if val_every > 0 and global_step % val_every == 0:
                    _run_validation(global_step)

                if eval_every > 0 and global_step % eval_every == 0:
                    _run_eval(global_step)

            log.info(
                "data pass %d done (step=%d/%d, train_loss=%.4f)",
                data_epoch,
                global_step,
                total_train_steps,
                last_loss,
            )

        if last_val_step != global_step:
            _run_validation(global_step, final=True)

        if last_ckpt_step != global_step:
            _save_checkpoint(global_step)
    finally:
        if eval_runner is not None:
            eval_runner.close()
        logger.close()

    log.info("BC training complete.")
    print("[bc_train] Training complete !!!", flush=True)
