"""Behavior Cloning fine-tune for any VLA registered in :mod:`vla.models`.

This is the **main** training entry point. It is called from the Hydra
top-level script ``scripts/train.py`` with ``mode=bc``.

Pipeline
--------
    1. Build the VLA via the registry (``cfg.model.name``).
    2. Load a ``LeRobotDataset`` produced by
       :mod:`data.hdf5_to_lerobot`.
    3. Standard supervised loop: mini-batch -> ``vla.compute_loss`` ->
       Adam/AdamW step -> log -> eval every ``cfg.mode.eval_every`` epochs.
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
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from vla.data.obs_adapter import ObsAdapterConfig
log = logging.getLogger(__name__)


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


def _ensure_local_episode_metadata(root: Path, fps: int) -> None:
    """Create missing ``meta/episodes`` metadata for a local LeRobot dataset."""
    episodes_dir = root / "meta" / "episodes"
    if any(episodes_dir.glob("*/*.parquet")):
        return

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
    for metadata_file_index, ep_idx in enumerate(sorted(episode_records)):
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
            "meta/episodes/file_index": metadata_file_index,
        }

        start_ts = row["dataset_from_index"] / fps
        end_ts = row["dataset_to_index"] / fps
        for video_key in video_keys:
            row[f"videos/{video_key}/chunk_index"] = 0
            row[f"videos/{video_key}/file_index"] = 0
            row[f"videos/{video_key}/from_timestamp"] = start_ts
            row[f"videos/{video_key}/to_timestamp"] = end_ts

        rows.append(row)

    episodes_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        out_path = episodes_dir / "chunk-000" / f"file-{row['meta/episodes/file_index']:03d}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([row]), out_path)

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

    # delta_timestamps lets LeRobot serve action chunks aligned with
    # SmolVLA's chunk_size; overridable per model via cfg.model.
    delta_timestamps = None
    if cfg.dataset.get("action_chunk_size", None):
        k = int(cfg.dataset.action_chunk_size)
        delta_timestamps = {"action": [i / cfg.dataset.fps for i in range(k)]}

    return LeRobotDataset(
        repo_id=cfg.dataset.repo_id,
        root=root,
        delta_timestamps=delta_timestamps,
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


# ---------------------------------------------------------------------
# Evaluation harness (Matterix-in-the-loop)
# ---------------------------------------------------------------------
def _build_evaluator(cfg: DictConfig, adapter_cfg: ObsAdapterConfig):
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

    # Build evaluator first so simulator startup happens at trainer start,
    # not later after dataset/model setup.
    adapter_cfg = ObsAdapterConfig(**OmegaConf.to_container(cfg.task.adapter, resolve=True))
    eval_fn = _build_evaluator(cfg, adapter_cfg)

    # -- Dataset -------------------------------------------------------
    log.info("Building training dataset")
    print("[bc_train] Building training dataset", flush=True)
    dataset = _build_dataset(cfg)
    log.info("Dataset ready")
    loader = DataLoader(
        dataset,
        batch_size=cfg.mode.batch_size,
        shuffle=True,
        num_workers=cfg.mode.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # -- Model ---------------------------------------------------------
    # Infer proprio state dim from a sample batch so we don't hardcode it.
    log.info("Loading sample batch to infer model dimensions")
    sample = next(iter(loader))
    state_dim = int(sample["observation.state"].shape[-1])
    action_dim = int(sample["action"].shape[-1])
    image_keys = [
        k.replace("observation.images.", "")
        for k in sample.keys()
        if k.startswith("observation.images.")
    ]

    log.info(
        "Building VLA model name=%s state_dim=%d action_dim=%d image_keys=%s",
        cfg.model.name,
        state_dim,
        action_dim,
        image_keys,
    )
    from vla.models import build_vla

    vla = build_vla(
        cfg.model.name,
        action_dim=action_dim,
        state_dim=state_dim,
        image_keys=image_keys,
        **OmegaConf.to_container(cfg.model.kwargs, resolve=True),
    )
    vla.to(device)
    log.info("Model ready on device=%s", device)

    # -- Optim + Logger ------------------------------------------------
    optim = _build_optimizer(vla, cfg)
    from vla.utils.logging import Logger

    logger = Logger(
        log_dir=Path(cfg.output_dir) / "tb",
        use_wandb=cfg.mode.get("use_wandb", False),
        wandb_project=cfg.mode.get("wandb_project", "labauto-vla"),
        wandb_run_name=cfg.mode.get("run_name", None),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # -- Loop ----------------------------------------------------------
    try:
        global_step = 0
        for epoch in range(cfg.mode.epochs):
            vla.train()
            log.info("Starting training epoch %d !!!", epoch + 1)
            for batch in loader:
                batch = vla.preprocess_batch(batch)
                log.info("Preprocess batch !!!")
                out = vla.compute_loss(batch)
                loss = out.loss
                if loss is None:
                    raise RuntimeError(f"{cfg.model.name}.compute_loss returned no loss")
                log.info("loss computed !!!")
                optim.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.mode.optim.grad_clip:
                    torch.nn.utils.clip_grad_norm_(vla.parameters(), cfg.mode.optim.grad_clip)
                optim.step()

                logger.scalar("train/loss", float(loss.detach()), global_step)
                global_step += 1

            log.info("epoch %d done (step=%d, loss=%.4f)", epoch, global_step, float(loss.detach()))

            # -- Periodic checkpoint + eval ---------------------------
            if (epoch + 1) % cfg.mode.ckpt_every == 0 or epoch + 1 == cfg.mode.epochs:
                ckpt_dir = Path(cfg.output_dir) / f"ckpt_epoch{epoch + 1:04d}"
                vla.save_pretrained(ckpt_dir)
                log.info("Saved checkpoint to %s", ckpt_dir)
                print("[bc_train] Saved checkpoint !!!", flush=True)

            if eval_fn is not None and (epoch + 1) % cfg.mode.eval_every == 0:
                print("[bc_train] Running evaluation !!!", flush=True)
                metrics = eval_fn(vla, step=epoch + 1)
                print("[bc_train] Evaluation complete !!!", flush=True)
                for k, v in metrics.items():
                    logger.scalar(f"eval/{k}", v, global_step)
                log.info("eval@epoch%d: %s", epoch + 1, metrics)
                print("[bc_train] eval@epoch%d: %s" % (epoch + 1, metrics), flush=True)
    finally:
        logger.close()

    log.info("BC training complete.")
    print("[bc_train] Training complete !!!", flush=True)
