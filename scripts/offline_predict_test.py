# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline teacher-forced action prediction test.

Loads a trained VLA checkpoint and produces two diagnostic plots:

1. offline_predict_test.png  — per-chunk-horizon analysis averaged over the
   validation split (same 10 % holdout as bc_train).

2. sequence_loss.png — per-timestep flow loss and step-0 L2 error plotted in
   temporal order over complete episodes from the FULL dataset (not the val
   split), so you see the 300-step task sequence as it unfolds.

No simulation is required — the test runs purely on the dataset.

Examples
--------
    python scripts/offline_predict_test.py
    python scripts/offline_predict_test.py n_samples=50
    python scripts/offline_predict_test.py n_seq_episodes=3
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _configure_logging() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _load_datasets(cfg: DictConfig):
    """Return (full_dataset, val_subset, chunk_size).

    full_dataset — the complete LeRobotDataset (used for the sequence plot)
    val_subset   — 10 % random split identical to bc_train's holdout
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError("pip install lerobot") from exc

    from vla.training.bc_train import _ensure_local_episode_metadata

    root = Path(cfg.dataset.root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset not found: {root}\nRun generate_dataset first."
        )

    _ensure_local_episode_metadata(root, fps=int(cfg.dataset.fps))

    chunk_size = int(cfg.dataset.get("action_chunk_size", 50))
    fps = int(cfg.dataset.fps)
    delta_timestamps = {"action": [i / fps for i in range(chunk_size)]}

    full_dataset = LeRobotDataset(
        repo_id=cfg.dataset.repo_id,
        root=root,
        delta_timestamps=delta_timestamps,
    )

    val_fraction = float(cfg.get("validation_fraction", 0.1))
    seed = int(cfg.get("split_seed", 42))
    n = len(full_dataset)
    n_val = max(1, int(round(n * val_fraction)))
    n_train = n - n_val
    generator = torch.Generator().manual_seed(seed)
    _, val_ds = random_split(full_dataset, [n_train, n_val], generator=generator)

    return full_dataset, val_ds, chunk_size


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _load_model(cfg: DictConfig, action_dim: int, state_dim: int, image_keys: list[str]) -> Any:
    from vla.models import build_vla

    model_kwargs = dict(OmegaConf.to_container(cfg.model.get("kwargs", {}), resolve=True))
    model_kwargs["device"] = cfg.device
    vla = build_vla(
        cfg.model.name,
        action_dim=action_dim,
        state_dim=state_dim,
        image_keys=image_keys,
        **model_kwargs,
    )
    vla.to(torch.device(cfg.device))
    vla.eval()
    return vla


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _image_keys_from_batch(batch: dict) -> list[str]:
    prefix = "observation.images."
    return [k[len(prefix):] for k in batch if k.startswith(prefix)]


def _predict_action_chunk(vla: Any, batch: dict, chunk_size: int) -> np.ndarray:
    """Teacher-forced prediction: one GT observation -> full predicted chunk.

    Patches n_action_steps to chunk_size so a single inference call fills
    the action buffer with all chunk_size steps, then drains it.

    Returns np.ndarray of shape (chunk_size, action_dim).
    """
    cfg_objects = []
    for attr in ("cfg", "policy"):
        obj = getattr(vla, attr, None)
        if obj is not None and hasattr(obj, "config"):
            cfg_objects.append(obj.config)
        elif obj is not None and hasattr(obj, "n_action_steps"):
            cfg_objects.append(obj)
    if hasattr(vla, "cfg") and hasattr(vla.cfg, "n_action_steps"):
        cfg_objects.append(vla.cfg)

    orig_values = [(c, c.n_action_steps) for c in cfg_objects if hasattr(c, "n_action_steps")]
    for c, _ in orig_values:
        c.n_action_steps = chunk_size

    try:
        vla.reset()
        image_keys = _image_keys_from_batch(batch)
        images = {}
        for k in image_keys:
            img = batch[f"observation.images.{k}"]
            if img.ndim == 5:
                img = img[:, 0]
            images[k] = img

        state = batch["observation.state"]
        if state.ndim == 3:
            state = state[:, 0]

        task = batch.get("task", [""])
        if not isinstance(task, list):
            task = list(task)

        predicted: list[np.ndarray] = []
        with torch.inference_mode():
            for _ in range(chunk_size):
                action = vla.predict_action(images, state, task)
                predicted.append(action.squeeze(0).cpu().float().numpy())
    finally:
        for c, orig in orig_values:
            c.n_action_steps = orig

    return np.stack(predicted, axis=0)  # (chunk_size, A)


def _step0_l2(vla: Any, batch: dict) -> float:
    """Single forward pass -> L2 error of the first predicted action vs GT."""
    vla.reset()
    image_keys = _image_keys_from_batch(batch)
    images = {}
    for k in image_keys:
        img = batch[f"observation.images.{k}"]
        if img.ndim == 5:
            img = img[:, 0]
        images[k] = img

    state = batch["observation.state"]
    if state.ndim == 3:
        state = state[:, 0]

    task = batch.get("task", [""])
    if not isinstance(task, list):
        task = list(task)

    with torch.inference_mode():
        pred = vla.predict_action(images, state, task)  # (1, A)

    gt = batch["action"]
    gt = gt[:, 0, :] if gt.ndim == 3 else gt   # (1, A)
    return float(torch.norm(pred.cpu().float() - gt.cpu().float()).item())


# ---------------------------------------------------------------------------
# Sequence analysis — full dataset in temporal order
# ---------------------------------------------------------------------------

def _run_sequence_analysis(
    vla: Any,
    full_dataset,
    cfg: DictConfig,
    n_episodes: int | None,
    log: logging.Logger,
) -> list[tuple[int, int, float, float]]:
    """Iterate complete episodes in temporal order and compute per-frame errors.

    LeRobot datasets are stored episode-by-episode in index order, so we
    simply iterate the full dataset and stop once we've covered n_episodes.

    Returns list of (episode_idx, frame_idx, flow_loss, step0_l2).
    """
    loader = DataLoader(
        full_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    records: list[tuple[int, int, float, float]] = []
    n_total = len(full_dataset)

    for i, batch in enumerate(loader):
        ep_idx = int(batch["episode_index"].item()) if "episode_index" in batch else 0
        fr_idx = int(batch["frame_index"].item())   if "frame_index"   in batch else i

        # Stop once we've passed the requested episode count
        if n_episodes is not None and ep_idx >= n_episodes:
            break

        # Flow-matching loss
        dev_batch = {
            k: v.to(cfg.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        preprocessed = vla.preprocess_batch(dev_batch)
        with torch.no_grad():
            loss_val = float(vla.compute_loss(preprocessed).loss.detach())

        l2 = _step0_l2(vla, batch)
        records.append((ep_idx, fr_idx, loss_val, l2))

        if (i + 1) % max(1, min(n_total, 500) // 10) == 0:
            log.info(
                "  seq [%d]  ep=%d fr=%d  loss=%.4f  L2=%.4f",
                i + 1, ep_idx, fr_idx, loss_val, l2,
            )

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="offline_predict")
def main(cfg: DictConfig) -> None:
    _configure_logging()
    log = logging.getLogger(__name__)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    log.info("Loading datasets...")
    full_dataset, val_ds, chunk_size = _load_datasets(cfg)

    # Infer model dimensions from one dataset sample (mirrors bc_train flow)
    _probe = val_ds[0]
    state_dim  = int(_probe["observation.state"].shape[-1])
    action_dim = int(_probe["action"].shape[-1])
    image_keys = [k[len("observation.images."):] for k in _probe if k.startswith("observation.images.")]
    log.info("Inferred  action_dim=%d  state_dim=%d  image_keys=%s", action_dim, state_dim, image_keys)

    log.info("Loading model...")
    vla = _load_model(cfg, action_dim=action_dim, state_dim=state_dim, image_keys=image_keys)

    # -----------------------------------------------------------------------
    # Pass 1: chunk-horizon analysis over val split
    # -----------------------------------------------------------------------
    n_total = len(val_ds)
    n_samples = n_total if cfg.get("n_samples") is None else min(int(cfg.n_samples), n_total)
    log.info("Val samples: %d  |  using: %d  |  chunk_size: %d", n_total, n_samples, chunk_size)

    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False,
    )

    per_step_l2  = np.zeros(chunk_size)
    per_step_mse = np.zeros(chunk_size)
    per_dim_mse: np.ndarray | None = None
    val_losses: list[float] = []

    for i, batch in enumerate(val_loader):
        if i >= n_samples:
            break

        dev_batch = {
            k: v.to(cfg.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        preprocessed = vla.preprocess_batch(dev_batch)
        with torch.no_grad():
            out = vla.compute_loss(preprocessed)
        val_losses.append(float(out.loss.detach()))

        pred = _predict_action_chunk(vla, batch, chunk_size)    # (T, A)
        gt   = batch["action"].squeeze(0).cpu().float().numpy() # (T, A)
        T = min(pred.shape[0], gt.shape[0])
        pred, gt = pred[:T], gt[:T]

        l2  = np.linalg.norm(pred - gt, axis=-1)
        mse = np.mean((pred - gt) ** 2, axis=-1)
        per_step_l2[:T]  += l2
        per_step_mse[:T] += mse

        if per_dim_mse is None:
            per_dim_mse = np.zeros((chunk_size, gt.shape[-1]))
        per_dim_mse[:T] += (pred - gt) ** 2

        if (i + 1) % max(1, n_samples // 10) == 0:
            log.info("  [%d/%d]  loss=%.4f  L2_step0=%.4f", i + 1, n_samples, val_losses[-1], l2[0])

    per_step_l2  /= len(val_losses)
    per_step_mse /= len(val_losses)
    per_dim_mse  /= len(val_losses)
    mean_loss = float(np.mean(val_losses))
    log.info("Mean val loss: %.4f  |  L2 step0: %.4f  |  L2 last: %.4f",
             mean_loss, float(per_step_l2[0]), float(per_step_l2[-1]))

    np.save(out_dir / "val_losses.npy",   np.array(val_losses))
    np.save(out_dir / "per_step_l2.npy",  per_step_l2)
    np.save(out_dir / "per_step_mse.npy", per_step_mse)
    np.save(out_dir / "per_dim_mse.npy",  per_dim_mse)

    # -----------------------------------------------------------------------
    # Pass 2: sequence analysis over the FULL dataset in temporal order
    # -----------------------------------------------------------------------
    n_seq_episodes = cfg.get("n_seq_episodes", None)
    if n_seq_episodes is not None:
        n_seq_episodes = int(n_seq_episodes)
        log.info("Running sequence analysis for first %d episode(s)...", n_seq_episodes)
    else:
        log.info("Running sequence analysis over all episodes...")

    seq_records = _run_sequence_analysis(vla, full_dataset, cfg, n_seq_episodes, log)
    seq_episodes = np.array([r[0] for r in seq_records], dtype=int)
    seq_frames   = np.array([r[1] for r in seq_records], dtype=int)
    seq_loss     = np.array([r[2] for r in seq_records])
    seq_l2       = np.array([r[3] for r in seq_records])

    np.save(out_dir / "sequence_loss.npy", seq_loss)
    np.save(out_dir / "sequence_l2.npy",   seq_l2)

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    import matplotlib.pyplot as plt

    ckpt_label = str(cfg.model.kwargs.get("pretrained_name_or_path", ""))
    steps = np.arange(chunk_size)
    adim = per_dim_mse.shape[-1]
    dim_labels = (
        ["x", "y", "z", "qw", "qx", "qy", "qz", "grip"][:adim]
        if adim == 8 else [f"d{d}" for d in range(adim)]
    )

    # --- Plot 1: chunk-horizon analysis ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Offline Action Prediction  —  {n_samples} val samples\nckpt: {ckpt_label}",
        fontsize=10,
    )

    ax = axes[0, 0]
    ax.plot(val_losses, linewidth=0.8, alpha=0.7, label="per-sample loss")
    ax.axhline(mean_loss, color="red", linestyle="--", linewidth=1.2,
               label=f"mean = {mean_loss:.4f}")
    ax.set_xlabel("Validation sample index")
    ax.set_ylabel("Flow-matching loss")
    ax.set_title("Validation Loss per Sample")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps, per_step_l2, color="steelblue")
    ax.fill_between(steps, 0, per_step_l2, alpha=0.15, color="steelblue")
    ax.set_xlabel("Action chunk step")
    ax.set_ylabel("Mean L2 error  ||pred - GT||₂")
    ax.set_title("Predicted vs GT: L2 Error over Chunk Horizon")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps, per_step_mse, color="darkorange")
    ax.fill_between(steps, 0, per_step_mse, alpha=0.15, color="darkorange")
    ax.set_xlabel("Action chunk step")
    ax.set_ylabel("Mean MSE")
    ax.set_title("Predicted vs GT: MSE over Chunk Horizon")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    cmap = plt.get_cmap("tab10")
    for d, label in enumerate(dim_labels):
        ax.plot(steps, per_dim_mse[:, d], label=label, color=cmap(d), linewidth=1.2, alpha=0.85)
    ax.set_xlabel("Action chunk step")
    ax.set_ylabel("MSE per dimension")
    ax.set_title("Per-Dimension MSE over Chunk Horizon")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = out_dir / "offline_predict_test.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # --- Plot 2: sequence loss over full task timeline ---
    x_pos = np.arange(len(seq_records))

    # Mark episode boundaries
    ep_boundaries: list[int] = [
        i for i in range(1, len(seq_episodes))
        if seq_episodes[i] != seq_episodes[i - 1]
    ]

    fig2, (ax_loss, ax_l2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ep_label = f"{len(set(seq_episodes.tolist()))} episode(s), {len(seq_records)} frames"
    fig2.suptitle(
        f"Loss / L2 Error over Full Task Sequence  —  {ep_label}\nckpt: {ckpt_label}",
        fontsize=10,
    )

    for ax, values, ylabel, title, color in (
        (ax_loss, seq_loss, "Flow-matching loss",           "Flow-Matching Loss per Timestep",  "steelblue"),
        (ax_l2,  seq_l2,   "Step-0 L2  ||pred - GT||₂",   "Step-0 L2 Error per Timestep",     "darkorange"),
    ):
        ax.plot(x_pos, values, linewidth=0.8, color=color, alpha=0.8)
        ax.axhline(float(np.mean(values)), color="red", linestyle="--", linewidth=1,
                   label=f"mean = {np.mean(values):.4f}")
        for b in ep_boundaries:
            ax.axvline(b, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    ax_l2.set_xlabel("Frame index (episode-by-episode, left to right)")
    # Annotate episode start positions on x-axis
    tick_pos  = [0] + ep_boundaries
    tick_labs = [f"ep{seq_episodes[p]}" for p in tick_pos]
    ax_l2.set_xticks(tick_pos)
    ax_l2.set_xticklabels(tick_labs, fontsize=7, rotation=45)

    plt.tight_layout()
    seq_plot_path = out_dir / "sequence_loss.png"
    plt.savefig(seq_plot_path, dpi=150)
    plt.close()

    log.info("Saved chunk plot    -> %s", plot_path)
    log.info("Saved sequence plot -> %s", seq_plot_path)
    log.info("Saved arrays        -> %s/", out_dir)
    print(f"[offline_predict_test] mean val loss:   {mean_loss:.4f}", flush=True)
    print(f"[offline_predict_test] chunk plot:      {plot_path}", flush=True)
    print(f"[offline_predict_test] sequence plot:   {seq_plot_path}", flush=True)


if __name__ == "__main__":
    os.chdir(_REPO_ROOT)
    main()
