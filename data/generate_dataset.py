"""Synthetic dataset generation for VLA training.

This script now runs as a two-stage pipeline:
1. Collect raw HDF5 demos during simulation.
2. Convert the collected HDF5 data into a LeRobot v3.0 dataset (parquet + MP4).

Usage:
    Collect HDF5 and convert to LeRobot:
        python -m data.generate_dataset

    Collect only:
        python -m data.generate_dataset stages.collect_hdf5=true stages.convert_lerobot=false

    Convert only from existing HDF5:
        python -m data.generate_dataset stages.collect_hdf5=false stages.convert_lerobot=true
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Add project root to path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vla.data.obs_adapter import ObsAdapterConfig
from vla.data.task_language import TaskLanguageTemplate

from data.hdf5_to_lerobot import convert as convert_hdf5_to_lerobot
from data.hdf5_writer import HDF5DatasetWriter

if TYPE_CHECKING:
    from isaaclab.app import AppLauncher

log = logging.getLogger(__name__)

_app_launcher: "AppLauncher | None" = None
simulation_app = None


@dataclass
class OutputLayout:
    """Resolved output locations for one generation run."""

    dataset_name: str
    root_dir: Path
    hdf5_dir: Path
    lerobot_dir: Path
    hdf5_paths: dict[str, Path]


@dataclass
class TaskSpec:
    """Resolved task alias/config pair used during one generation run."""

    alias: str
    config_path: Path
    task_id: str
    cfg: DictConfig


# =============================================================================
# Utilities
# =============================================================================


def ensure_sim_app():
    """Launch Isaac Sim lazily so convert-only runs stay lightweight."""
    global _app_launcher, simulation_app
    if simulation_app is None:
        from isaaclab.app import AppLauncher

        _app_launcher = AppLauncher(headless=True, enable_cameras=True, livestream=2)
        simulation_app = _app_launcher.app
    return simulation_app


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for dataset generation."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    )


def set_seed(seed: int) -> None:
    """Set global random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    log.info("Global seed set to %d", seed)


def derive_dataset_name(cfg: DictConfig) -> str:
    """Determine the dataset name from config."""
    dataset_name = cfg.output.get("dataset_name")
    if dataset_name:
        return str(dataset_name)

    if len(cfg.generation.tasks) == 1:
        return f"{cfg.generation.tasks[0]}_v1"
    return "multi_task_v1"


def _sanitize_path_component(value: str) -> str:
    """Turn task IDs into safe file names."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return sanitized or "dataset"


def resolve_output_layout(cfg: DictConfig) -> OutputLayout:
    """Resolve all file-system paths used by collection and conversion."""
    root_dir = Path(cfg.output.root)
    dataset_name = derive_dataset_name(cfg)
    hdf5_dir = root_dir / cfg.output.hdf5_subdir / dataset_name
    lerobot_dir = root_dir / dataset_name
    hdf5_paths = {
        task_id: hdf5_dir / f"{_sanitize_path_component(task_id)}.hdf5"
        for task_id in cfg.generation.tasks
    }

    log.info("Dataset name: %s", dataset_name)
    log.info("HDF5 directory: %s", hdf5_dir)
    log.info("LeRobot directory: %s", lerobot_dir)
    return OutputLayout(
        dataset_name=dataset_name,
        root_dir=root_dir,
        hdf5_dir=hdf5_dir,
        lerobot_dir=lerobot_dir,
        hdf5_paths=hdf5_paths,
    )


def validate_stage_selection(cfg: DictConfig) -> None:
    """Ensure at least one pipeline stage is enabled."""
    if not cfg.stages.collect_hdf5 and not cfg.stages.convert_lerobot:
        raise ValueError("At least one stage must be enabled: collect_hdf5 or convert_lerobot.")


def resolve_task_specs(cfg: DictConfig, tasks: list[str]) -> dict[str, TaskSpec]:
    """Resolve requested task aliases into task configs and canonical env IDs."""
    configured_paths = cfg.generation.get("task_cfg_paths")
    if not configured_paths:
        raise ValueError("generation.task_cfg_paths must define at least one task config path.")

    task_specs: dict[str, TaskSpec] = {}
    for task_alias in tasks:
        if task_alias not in configured_paths:
            available = ", ".join(sorted(str(name) for name in configured_paths.keys()))
            raise KeyError(
                f"Task '{task_alias}' is not defined in generation.task_cfg_paths. "
                f"Available tasks: {available}"
            )

        config_path = Path(str(configured_paths[task_alias]))
        if not config_path.is_absolute():
            config_path = _REPO_ROOT / config_path

        if not config_path.exists():
            raise FileNotFoundError(f"Task config for '{task_alias}' was not found: {config_path}")

        task_cfg = OmegaConf.load(config_path)
        task_id = task_cfg.get("id")
        if not task_id:
            raise ValueError(f"Task config '{config_path}' is missing required 'id' field.")

        task_specs[task_alias] = TaskSpec(
            alias=task_alias,
            config_path=config_path,
            task_id=str(task_id),
            cfg=task_cfg,
        )

    return task_specs


def create_language_generators(
    task_specs: dict[str, TaskSpec], seed: int | None = None
) -> dict[str, TaskLanguageTemplate]:
    """Create language generators for each task."""
    generators = {}

    for task_alias, task_spec in task_specs.items():
        task_cfg = task_spec.cfg
        if "language_generation" not in task_cfg:
            log.warning(
                "Task %s has no language_generation config. Using default prompt.", task_alias
            )
            task_cfg_dict = {
                "templates": [task_cfg.get("prompt", f"Perform {task_alias} task")],
                "parameters": {},
            }
        else:
            task_cfg_dict = OmegaConf.to_container(task_cfg.language_generation, resolve=True)

        generators[task_alias] = TaskLanguageTemplate(
            task_name=task_alias,
            config=task_cfg_dict,
            seed=seed,
        )

    log.info("Created language generators for %d tasks", len(generators))
    return generators


def _adapter_cfg_from_task_spec(task_spec: TaskSpec) -> ObsAdapterConfig:
    """Build the VLA observation adapter declared by a task config."""
    adapter = task_spec.cfg.get("adapter")
    if adapter is None:
        log.warning("Task %s has no adapter config. Using default overhead-only adapter.", task_spec.alias)
        return ObsAdapterConfig()

    adapter_dict = OmegaConf.to_container(adapter, resolve=True)
    if not isinstance(adapter_dict, dict):
        raise TypeError(f"Task {task_spec.alias} adapter config must be a mapping.")

    default_cfg = ObsAdapterConfig()
    image_keys = dict(adapter_dict.get("image_keys", default_cfg.image_keys))
    state_keys = list(adapter_dict.get("state_keys", default_cfg.state_keys))
    image_size = tuple(adapter_dict.get("image_size", default_cfg.image_size))

    if len(image_size) != 2:
        raise ValueError(f"Task {task_spec.alias} adapter.image_size must be [height, width].")
    if not image_keys:
        raise ValueError(f"Task {task_spec.alias} adapter.image_keys must contain at least one camera.")

    return ObsAdapterConfig(
        image_keys=image_keys,
        state_keys=state_keys,
        image_size=(int(image_size[0]), int(image_size[1])),
    )


def resolve_adapter_cfg(cfg: DictConfig) -> ObsAdapterConfig:
    """Resolve the single LeRobot schema adapter for the selected tasks."""
    task_specs = resolve_task_specs(cfg, cfg.generation.tasks)
    adapters = {
        task_alias: _adapter_cfg_from_task_spec(task_spec)
        for task_alias, task_spec in task_specs.items()
    }

    first_alias = str(cfg.generation.tasks[0])
    first_adapter = adapters[first_alias]
    for task_alias, adapter in adapters.items():
        if adapter != first_adapter:
            raise ValueError(
                "All tasks in one LeRobot dataset must use the same adapter schema. "
                f"Task '{first_alias}' uses {first_adapter}, but task '{task_alias}' uses {adapter}."
            )

    log.info("LeRobot adapter image keys: %s", first_adapter.image_keys)
    log.info("LeRobot adapter state keys: %s", first_adapter.state_keys)
    return first_adapter


def create_task_runtimes(
    cfg: DictConfig,
    task_aliases: list[str],
    task_specs: dict[str, TaskSpec],
) -> dict[str, dict[str, Any]]:
    """Create environments and workflow executors for each task."""
    ensure_sim_app()

    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    import matterix_tasks  # noqa: F401 - register envs
    from data.workflow_executor import create_workflow_executor

    task_runtimes: dict[str, dict[str, Any]] = {}

    for task_alias in task_aliases:
        task_spec = task_specs[task_alias]
        task_id = task_spec.task_id
        log.info("Creating environment for task: %s -> %s", task_alias, task_id)

        env_cfg = parse_env_cfg(
            task_id,
            device=cfg.generation.device,
            num_envs=cfg.generation.num_envs,
        )
        env = gym.make(task_id, cfg=env_cfg).unwrapped

        preferred_workflow_name = None
        task_cfg = task_spec.cfg
        if task_cfg is not None and "workflow" in task_cfg and "name" in task_cfg.workflow:
            preferred_workflow_name = str(task_cfg.workflow.name)

        workflow_executor = create_workflow_executor(
            env=env,
            env_cfg=env_cfg,
            preferred_workflow_name=preferred_workflow_name,
            suppress_output=bool(cfg.logging.get("suppress_workflow_output", True)),
        )

        task_runtimes[task_alias] = {
            "env": env,
            "workflow_executor": workflow_executor,
            "task_id": task_id,
        }

        log.info("  num_envs: %s, device: %s", env.num_envs, cfg.generation.device)
        log.info("  workflow: %s", workflow_executor.workflow_name)

    return task_runtimes


def _env_done(done_flag: Any, env_index: int) -> bool:
    """Return done status for one environment from tensor/array/scalar flags."""
    if isinstance(done_flag, torch.Tensor):
        if done_flag.ndim == 0:
            return bool(done_flag.item())
        return bool(done_flag[env_index].item())
    if isinstance(done_flag, np.ndarray):
        if done_flag.ndim == 0:
            return bool(done_flag.item())
        return bool(done_flag[env_index])
    return bool(done_flag)


def _extract_env_tree(value: Any, env_index: int) -> Any:
    """Extract one env from nested observation trees."""
    if isinstance(value, dict):
        return {key: _extract_env_tree(sub_value, env_index) for key, sub_value in value.items()}
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return tensor[env_index].clone() if tensor.ndim > 0 else tensor.clone()
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        return np.array(array[env_index], copy=True) if array.ndim > 0 else np.array(array, copy=True)
    return np.asarray(value)


def _drop_recorded_robot_root_pose(obs: dict[str, Any]) -> dict[str, Any]:
    """Remove fixed robot root pose from observations written to HDF5."""
    articulations = obs.get("articulations")
    if isinstance(articulations, dict):
        articulations.pop("robot__root_world_pos", None)
        articulations.pop("robot__root_world_quat", None)
    return obs


_DEFAULT_PROPRIO_KEYS: list[str] = [
    "articulations/robot__ee_world_pos",
    "articulations/robot__ee_world_quat",
    "articulations/robot__gripper_pos",
]


def _extract_proprio(obs: dict[str, Any], state_keys: list[str]) -> torch.Tensor:
    """Concatenate proprio state for env-0 from a raw multi-env obs dict.

    Traverses slash-separated paths (e.g. ``"articulations/robot__ee_world_pos"``)
    and returns a flat float32 tensor of the concatenated values for env index 0.
    """
    parts: list[torch.Tensor] = []
    for key in state_keys:
        node: Any = obs
        for part in key.split("/"):
            node = node[part]
        if isinstance(node, torch.Tensor):
            vec = node[0].detach().cpu().float().flatten()
        else:
            vec = torch.tensor(np.asarray(node)[0], dtype=torch.float32).flatten()
        parts.append(vec)
    return torch.cat(parts, dim=0)


def _extract_next_ee_actions_base(obs: dict[str, Any]) -> torch.Tensor:
    """Return achieved next EE poses in the env's 8D base-frame action layout.

    The current pipetting robot base is fixed at the world origin, so world EE
    pose and base-frame EE pose are equivalent.
    """

    artic = obs["articulations"]

    ee_pos_w = artic["robot__ee_world_pos"]
    ee_quat_w = artic["robot__ee_world_quat"]

    finger_pos = artic["robot__gripper_pos"]  # (N, 2)
    gripper_open = (
        finger_pos.abs().clamp(0.0, 0.04).mean(dim=-1, keepdim=True) / 0.04
    )

    return torch.cat([ee_pos_w, ee_quat_w, gripper_open], dim=-1).detach().cpu().float()


def _stack_tree(values: list[Any]) -> Any:
    """Stack a list of nested tensors/arrays into a trajectory tree."""
    first = values[0]
    if isinstance(first, dict):
        return {
            key: _stack_tree([value[key] for value in values])
            for key in first
        }
    if isinstance(first, torch.Tensor):
        return torch.stack(values, dim=0)
    return np.stack([np.asarray(value) for value in values], axis=0)


def extract_domain_state(env: Any, task_id: str) -> dict[str, Any]:
    """Extract randomized scene state from environment after reset."""
    domain_state = {}

    try:
        scene = env.scene

        if task_id == "pipetting" or "pipetting" in task_id.lower():
            if "source_beaker" in scene:
                domain_state["source_pos"] = scene["source_beaker"].data.root_pos_w[0].cpu().numpy()
            if "target_beaker" in scene:
                domain_state["target_pos"] = scene["target_beaker"].data.root_pos_w[0].cpu().numpy()

            domain_state["depth_cm"] = float(np.random.uniform(0.5, 3.0))

    except Exception as exc:
        log.debug("Could not extract domain state: %s", exc)

    return domain_state


def execute_episode(
    env: Any,
    workflow_executor: Any,
    task_id: str,
    task_prompt: str,
    cfg: DictConfig,
    max_steps: int | None = None,
    proprio_keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Execute vectorized episodes and return one HDF5-ready demo per env.

    Actions are recorded as the achieved next EE pose in robot base frame,
    plus achieved gripper opening. This gives dense 8D targets matching the
    env's pose-action layout without using the state machine's constant
    waypoint command directly.
    """
    if max_steps is None:
        max_steps = cfg.generation.max_steps
    if proprio_keys is None:
        proprio_keys = _DEFAULT_PROPRIO_KEYS

    num_envs = int(env.num_envs)
    raw_obs_steps: list[list[dict[str, Any]]] = [[] for _ in range(num_envs)]
    raw_action_steps: list[list[torch.Tensor]] = [[] for _ in range(num_envs)]
    finished = [False] * num_envs
    successes = [False] * num_envs
    obs, _ = env.reset()

    workflow_executor.reset()

    for _ in range(max_steps):
        active_envs = [env_idx for env_idx, done in enumerate(finished) if not done]
        if not active_envs:
            break

        for env_idx in active_envs:
            raw_obs_steps[env_idx].append(
                _drop_recorded_robot_root_pose(_extract_env_tree(obs, env_idx))
            )

        action_tensor = workflow_executor.step(obs)
        obs, _, terminated, truncated, _ = env.step(action_tensor)
        next_actions = _extract_next_ee_actions_base(obs)

        for env_idx in active_envs:
            # Record where the robot actually ended up after this step. Using the
            # state machine's goal directly produces constant actions per primitive.
            raw_action_steps[env_idx].append(next_actions[env_idx])

            workflow_done = workflow_executor.is_done(env_index=env_idx)
            env_done = _env_done(terminated, env_idx) or _env_done(truncated, env_idx)
            if workflow_done or env_done:
                if env_done and not workflow_done:
                    workflow_executor.mark_failed(env_idx)
                finished[env_idx] = True
                successes[env_idx] = workflow_executor.succeeded(env_index=env_idx)

    if not any(raw_action_steps):
        raise RuntimeError(f"Episode for task {task_id} produced no steps.")

    episodes: list[dict[str, Any]] = []
    for env_idx in range(num_envs):
        if not raw_action_steps[env_idx]:
            continue
        if not successes[env_idx]:
            log.warning(
                "Episode ended without workflow success. task=%s, env=%d, action_idx=%s, max_steps=%s",
                task_id,
                env_idx,
                workflow_executor.current_action_index(env_idx),
                max_steps,
            )

        episodes.append({
            "observations": _stack_tree(raw_obs_steps[env_idx]),
            "actions": torch.stack(raw_action_steps[env_idx], dim=0),
            "success": successes[env_idx],
            "length": len(raw_action_steps[env_idx]),
            "task_id": task_id,
            "task_prompt": task_prompt,
            "env_index": env_idx,
        })

    return episodes


def determine_episodes_per_task(cfg: DictConfig) -> dict[str, int]:
    """Resolve the target episode count for each task."""
    if cfg.generation.episodes_per_task:
        return {
            str(task_id): int(count)
            for task_id, count in cfg.generation.episodes_per_task.items()
        }

    episodes_per_task = {
        task: cfg.generation.num_episodes // len(cfg.generation.tasks)
        for task in cfg.generation.tasks
    }
    episodes_per_task[cfg.generation.tasks[0]] += (
        cfg.generation.num_episodes % len(cfg.generation.tasks)
    )
    return episodes_per_task


def collect_hdf5_datasets(cfg: DictConfig, layout: OutputLayout) -> dict[str, int]:
    """Collect raw episodes and write them to per-task HDF5 files."""
    task_specs = resolve_task_specs(cfg, cfg.generation.tasks)
    language_gens = create_language_generators(task_specs, seed=cfg.generation.seed)
    task_runtimes: dict[str, dict[str, Any]] = {}
    hdf5_writers: dict[str, HDF5DatasetWriter] = {}

    episodes_per_task = determine_episodes_per_task(cfg)
    log.info("Episodes per task: %s", episodes_per_task)

    episodes_collected = {task: 0 for task in cfg.generation.tasks}
    pbar = tqdm(total=cfg.generation.num_episodes, desc="Collecting HDF5 episodes")

    try:
        task_runtimes = create_task_runtimes(cfg, cfg.generation.tasks, task_specs)
        hdf5_writers = {
            task_alias: HDF5DatasetWriter(
                layout.hdf5_paths[task_alias],
                overwrite=bool(cfg.output.get("overwrite_hdf5", False)),
            )
            for task_alias in cfg.generation.tasks
        }

        task_queue: list[str] = []
        for task_alias, count in episodes_per_task.items():
            task_queue.extend([task_alias] * count)

        task_idx = 0
        while sum(episodes_collected.values()) < cfg.generation.num_episodes:
            task_alias = task_queue[task_idx % len(task_queue)]
            task_idx += 1
            if episodes_collected[task_alias] >= episodes_per_task[task_alias]:
                continue

            runtime = task_runtimes[task_alias]
            env = runtime["env"]
            workflow_executor = runtime["workflow_executor"]
            task_id = runtime["task_id"]

            domain_state = extract_domain_state(env, task_id)
            task_prompt = language_gens[task_alias].generate(**domain_state)

            task_spec = task_specs[task_alias]
            proprio_keys = list(
                OmegaConf.to_container(task_spec.cfg.adapter.state_keys, resolve=True)
            ) if "adapter" in task_spec.cfg and "state_keys" in task_spec.cfg.adapter else None

            try:
                episode_batch = execute_episode(
                    env=env,
                    workflow_executor=workflow_executor,
                    task_id=task_id,
                    task_prompt=task_prompt,
                    cfg=cfg,
                    proprio_keys=proprio_keys,
                )

                saved_this_rollout = 0
                for episode_data in episode_batch:
                    if episodes_collected[task_alias] >= episodes_per_task[task_alias]:
                        break
                    if sum(episodes_collected.values()) >= cfg.generation.num_episodes:
                        break
                    if cfg.generation.only_successful and not episode_data["success"]:
                        continue

                    hdf5_writers[task_alias].add_episode(
                        observations=episode_data["observations"],
                        actions=episode_data["actions"],
                        task=episode_data["task_prompt"],
                        task_id=task_id,
                        success=episode_data["success"],
                    )

                    episodes_collected[task_alias] += 1
                    saved_this_rollout += 1
                    pbar.update(1)

                if saved_this_rollout == 0:
                    continue

                if sum(episodes_collected.values()) % cfg.logging.progress_interval == 0:
                    log.info("Episodes collected: %s", episodes_collected)

            except Exception as exc:
                log.exception("Episode execution failed: %s", exc)
                continue
    finally:
        pbar.close()
        for writer in hdf5_writers.values():
            writer.close()
        for runtime in task_runtimes.values():
            runtime["env"].close()

    return episodes_collected


def write_generation_metadata(
    layout: OutputLayout,
    cfg: DictConfig,
    episodes_collected: dict[str, int] | None,
) -> None:
    """Write a small metadata file next to the converted LeRobot dataset."""
    if not layout.lerobot_dir.exists():
        return

    task_specs = resolve_task_specs(cfg, cfg.generation.tasks)
    adapter_cfg = resolve_adapter_cfg(cfg)
    metadata = {
        "dataset_name": layout.dataset_name,
        "hdf5_paths": {task_alias: str(path) for task_alias, path in layout.hdf5_paths.items()},
        "lerobot_dir": str(layout.lerobot_dir),
        "episodes_collected": episodes_collected or {},
        "stages": OmegaConf.to_container(cfg.stages, resolve=True),
        "tasks": list(cfg.generation.tasks),
        "resolved_task_ids": {
            task_alias: task_specs[task_alias].task_id for task_alias in cfg.generation.tasks
        },
        "fps": int(cfg.output.fps),
        "video_codec": str(cfg.output.video_codec),
        "video_crf": int(cfg.output.video_crf),
        "use_videos": bool(cfg.output.use_videos),
        "adapter": {
            "image_keys": adapter_cfg.image_keys,
            "state_keys": adapter_cfg.state_keys,
            "image_size": list(adapter_cfg.image_size),
        },
    }

    metadata_path = layout.lerobot_dir / "generation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def convert_hdf5_datasets(cfg: DictConfig, layout: OutputLayout) -> Path:
    """Convert collected HDF5 files into a LeRobot dataset."""
    src_paths = [layout.hdf5_paths[task_id] for task_id in cfg.generation.tasks]
    missing_paths = [path for path in src_paths if not path.exists()]
    if missing_paths:
        missing_str = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing HDF5 files for conversion: {missing_str}")

    repo_id = cfg.output.get("repo_id") or f"local/{layout.dataset_name}"
    adapter_cfg = resolve_adapter_cfg(cfg)
    return convert_hdf5_to_lerobot(
        src=src_paths,
        dst=layout.lerobot_dir,
        task=None,
        fps=cfg.output.fps,
        repo_id=repo_id,
        adapter_cfg=adapter_cfg,
        use_videos=cfg.output.use_videos,
        vcodec=cfg.output.video_codec,
        video_crf=int(cfg.output.video_crf),
        frames_per_chunk=int(cfg.output.get("frames_per_chunk", 64)),
        overwrite=bool(cfg.output.get("overwrite_lerobot", False)),
    )


def run_generation(cfg: DictConfig) -> None:
    """Main dataset generation/conversion entry point."""
    setup_logging(cfg.logging.level)
    set_seed(cfg.generation.seed)
    validate_stage_selection(cfg)

    log.info("=" * 80)
    log.info("LabAutoVLA Synthetic Dataset Generation")
    log.info("=" * 80)
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    layout = resolve_output_layout(cfg)
    episodes_collected: dict[str, int] | None = None

    if cfg.stages.collect_hdf5:
        log.info("Stage 1/2: Collecting HDF5 demos")
        episodes_collected = collect_hdf5_datasets(cfg, layout)
        log.info("HDF5 collection complete: %s", episodes_collected)

    if cfg.stages.convert_lerobot:
        log.info("Stage 2/2: Converting HDF5 to LeRobot v3.0")
        convert_hdf5_datasets(cfg, layout)
        write_generation_metadata(layout, cfg, episodes_collected)

    log.info("=" * 80)
    log.info("Dataset Pipeline Complete")
    log.info("=" * 80)
    log.info("HDF5 directory: %s", layout.hdf5_dir)
    log.info("LeRobot directory: %s", layout.lerobot_dir)
    if episodes_collected is not None:
        log.info("Episodes collected: %s", episodes_collected)


@hydra.main(version_base=None, config_path="../configs", config_name="generation")
def main(cfg: DictConfig) -> None:
    """Entry point for Hydra."""
    try:
        run_generation(cfg)
    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()
