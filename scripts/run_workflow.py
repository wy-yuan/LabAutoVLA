# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Run a workflow using the MATteRIX state machine system.

The state machine orchestrates sequential actions across parallel environments on the GPU.

Usage:
    ./matterix.sh -p scripts/run_workflow.py --task Matterix-Test-Beaker-Lift-Franka-v1 --workflow pickup_beaker
    ./matterix.sh -p scripts/run_workflow.py --num_envs 32
"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Run state machine workflows for MATteRIX environments.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of parallel environments to simulate.",
)
parser.add_argument(
    "--task",
    type=str,
    default="Matterix-Test-Beaker-Lift-Franka-v1",
    help="Environment/task name.",
)
parser.add_argument("--workflow", type=str, default="pickup_beaker", help="Name of the workflow to run.")
parser.add_argument(
    "--save_snapshots",
    action="store_true",
    default=False,
    help="Save periodic camera snapshots (requires --enable_cameras).",
)
parser.add_argument(
    "--snapshot_interval",
    type=int,
    default=20,
    help="Save a snapshot every N env steps when --save_snapshots is set.",
)
parser.add_argument(
    "--snapshot_dir",
    type=str,
    default="snapshots",
    help="Root directory for saved snapshots.",
)
parser.add_argument(
    "--snapshot_camera",
    type=str,
    default="overhead_camera",
    help="Scene key of the TiledCamera to snapshot.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app
# app_launcher = AppLauncher(headless=args_cli.headless)
# Pass full args_cli so flags like --enable_cameras propagate to the launcher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import datetime as _dt
import os

import gymnasium as gym
import torch
from PIL import Image

import matterix_tasks  # noqa: F401
from matterix_sm import StateMachine

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def _save_camera_snapshot(env, camera_key: str, out_dir: str, episode: int, step: int):
    """Save RGB frames from a TiledCamera to ``out_dir``.

    Saves one PNG per parallel env: ``episode_{e}_step_{s}_env_{n}.png``.
    The camera tensor is (num_envs, H, W, 3 or 4). We accept either raw uint8
    from ``camera.data.output["rgb"]`` or a float [0, 1] tensor from the
    observation function, and normalise to uint8 for saving.
    """
    camera = env.scene[camera_key]
    rgb = camera.data.output["rgb"]  # (N, H, W, 3 or 4)
    rgb = rgb[..., :3]  # drop alpha if present

    if rgb.dtype.is_floating_point:
        rgb = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8)

    rgb_np = rgb.detach().cpu().numpy()
    os.makedirs(out_dir, exist_ok=True)

    for env_idx in range(rgb_np.shape[0]):
        path = os.path.join(out_dir, f"episode_{episode:03d}_step_{step:05d}_env_{env_idx}.png")
        Image.fromarray(rgb_np[env_idx]).save(path)


def _make_snapshot_dir(root: str) -> str:
    """Create a timestamped snapshot folder so iterations don't overwrite."""
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(root, ts)
    os.makedirs(out, exist_ok=True)
    return out


def main():
    # Parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # Validate workflow exists
    if not hasattr(env_cfg, "workflows") or not env_cfg.workflows:
        raise ValueError(f"No workflows defined for {args_cli.task}!")

    if args_cli.workflow not in env_cfg.workflows:
        available = list(env_cfg.workflows.keys())
        raise ValueError(
            f"Workflow '{args_cli.workflow}' not found. Available workflows: {available}. "
            f"Use 'python scripts/list_workflows.py --task {args_cli.task}' to see details."
        )

    # Extract workflow actions
    workflow_value = env_cfg.workflows[args_cli.workflow]
    if isinstance(workflow_value, dict):
        description = workflow_value.get("description", "No description")
        actions = workflow_value.get("actions", [])
    else:
        description = getattr(workflow_value, "description", "No description")
        actions = [workflow_value]

    print(f"\nTask: {args_cli.task}")
    print(f"Workflow: '{args_cli.workflow}'")
    print(f"Description: {description}\n")

    # Create environment and state machine
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    # Create state machine with required parameters from environment
    sm = StateMachine(num_envs=env.num_envs, dt=env.step_dt, device=env.device)
    sm.set_action_sequence(actions)

    # Set up camera snapshot output directory (timestamped so iterations don't clash)
    snapshot_dir = None
    if args_cli.save_snapshots:
        if args_cli.snapshot_camera not in env.scene.keys():
            raise KeyError(
                f"Camera '{args_cli.snapshot_camera}' not found in scene. "
                f"Available keys: {list(env.scene.keys())}"
            )
        snapshot_dir = _make_snapshot_dir(args_cli.snapshot_dir)
        print(f"[INFO] Saving camera snapshots every {args_cli.snapshot_interval} steps to: {snapshot_dir}")

    episode_count = 0

    # Main simulation loop
    while simulation_app.is_running():
        with torch.inference_mode():
            obs, _ = env.reset()
            sm.reset()
            episode_count += 1
            step_count = 0

            print(f"\n{'=' * 80}")
            print(f"EPISODE {episode_count}")
            print(f"{'=' * 80}\n")

            # Snapshot at the start of the episode (post-reset view)
            if snapshot_dir is not None:
                _save_camera_snapshot(
                    env, args_cli.snapshot_camera, snapshot_dir, episode_count, step_count
                )

            # Run until workflow completes or fails
            while not (sm.action_sequence_success | sm.action_sequence_failure).all():
                action = sm.step(obs).to(env.device)
                obs, _, _, _, _ = env.step(action)
                step_count += 1

                # Print status every 50 steps
                if step_count % 50 == 0:
                    sm.print_status(step=step_count, episode=episode_count)

                # Periodic camera snapshots
                if snapshot_dir is not None and step_count % args_cli.snapshot_interval == 0:
                    _save_camera_snapshot(
                        env, args_cli.snapshot_camera, snapshot_dir, episode_count, step_count
                    )

            # Episode finished
            sm.print_status(step=step_count, episode=episode_count)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
