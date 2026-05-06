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
    default=None,
    help=(
        "Comma-separated scene key(s) of the TiledCamera(s) to snapshot and record. "
        "Defaults to overhead_camera and wrist_camera when present."
    ),
)
parser.add_argument(
    "--save_video",
    action="store_true",
    default=False,
    help="Save a video per episode from the snapshot camera(s) (requires --enable_cameras).",
)
parser.add_argument(
    "--video_fps",
    type=int,
    default=30,
    help="Frames per second for the saved video.",
)
parser.add_argument(
    "--video_episodes",
    type=int,
    default=1,
    help="Number of episodes to record (starting from the first).",
)
parser.add_argument(
    "--video_stride",
    type=int,
    default=1,
    help="Record a frame every N env steps (1 = every step).",
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
import imageio.v2 as imageio
import torch
from PIL import Image

import matterix_tasks  # noqa: F401
from matterix_sm import StateMachine

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


DEFAULT_CAMERA_KEYS = ("overhead_camera", "wrist_camera")


def _parse_camera_keys(camera_arg: str) -> list[str]:
    """Parse a comma-separated camera key argument."""
    camera_keys = [key.strip() for key in camera_arg.split(",") if key.strip()]
    if not camera_keys:
        raise ValueError("--snapshot_camera must contain at least one camera key.")
    return camera_keys


def _resolve_camera_keys(env, camera_arg: str | None) -> list[str]:
    """Resolve cameras to record, defaulting to known workflow cameras when available."""
    if camera_arg is not None:
        return _parse_camera_keys(camera_arg)

    scene_keys = set(env.scene.keys())
    camera_keys = [camera_key for camera_key in DEFAULT_CAMERA_KEYS if camera_key in scene_keys]
    if not camera_keys:
        raise KeyError(
            "No default cameras found in scene. "
            f"Looked for {list(DEFAULT_CAMERA_KEYS)}. Available keys: {list(env.scene.keys())}"
        )
    return camera_keys


def _save_camera_snapshot(env, camera_key: str, out_dir: str, episode: int, step: int):
    """Save RGB frames from a TiledCamera to ``out_dir``.

    Saves one PNG per parallel env under ``out_dir/{camera_key}``:
    ``episode_{e}_step_{s}_env_{n}.png``.
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
    camera_dir = os.path.join(out_dir, camera_key)
    os.makedirs(camera_dir, exist_ok=True)

    for env_idx in range(rgb_np.shape[0]):
        path = os.path.join(camera_dir, f"episode_{episode:03d}_step_{step:05d}_env_{env_idx}.png")
        Image.fromarray(rgb_np[env_idx]).save(path)


def _make_snapshot_dir(root: str) -> str:
    """Create a timestamped snapshot folder so iterations don't overwrite."""
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(root, ts)
    os.makedirs(out, exist_ok=True)
    return out


def _grab_camera_frames(env, camera_key: str):
    """Return a uint8 RGB numpy array of shape (num_envs, H, W, 3)."""
    camera = env.scene[camera_key]
    rgb = camera.data.output["rgb"][..., :3]
    if rgb.dtype.is_floating_point:
        rgb = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    return rgb.detach().cpu().numpy()


def _open_episode_writers(out_dir: str, camera_key: str, episode: int, num_envs: int, fps: int):
    """Open one mp4 writer per parallel env for the given episode."""
    camera_dir = os.path.join(out_dir, camera_key)
    os.makedirs(camera_dir, exist_ok=True)
    writers = []
    for env_idx in range(num_envs):
        path = os.path.join(camera_dir, f"episode_{episode:03d}_env_{env_idx}.mp4")
        writers.append(imageio.get_writer(path, fps=fps, codec="libx264", quality=8))
    return writers


def main():
    # Parse configuration
    parse_cfg_kwargs = {
        "device": args_cli.device,
        "num_envs": args_cli.num_envs,
    }
    # Let particle-enabled env configs keep their own fabric setting. Forcing
    # use_fabric=True breaks particle rendering/simulation in the pipetting task.
    if args_cli.disable_fabric:
        parse_cfg_kwargs["use_fabric"] = False

    env_cfg = parse_env_cfg(args_cli.task, **parse_cfg_kwargs)
    print(
        "[INFO] Parsed env config: "
        f"enable_particles={getattr(env_cfg, 'enable_particles', False)}, "
        f"use_fabric={getattr(env_cfg.sim, 'use_fabric', 'unknown')}"
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
    video_dir = None
    camera_keys = []
    if args_cli.save_snapshots or args_cli.save_video:
        camera_keys = _resolve_camera_keys(env, args_cli.snapshot_camera)
        missing_camera_keys = [camera_key for camera_key in camera_keys if camera_key not in env.scene.keys()]
        if missing_camera_keys:
            raise KeyError(
                f"Camera(s) {missing_camera_keys} not found in scene. "
                f"Available keys: {list(env.scene.keys())}"
            )
    if args_cli.save_snapshots:
        snapshot_dir = _make_snapshot_dir(args_cli.snapshot_dir)
        print(
            f"[INFO] Saving camera snapshots from {camera_keys} "
            f"every {args_cli.snapshot_interval} steps to: {snapshot_dir}"
        )
    if args_cli.save_video:
        video_dir = _make_snapshot_dir(os.path.join(args_cli.snapshot_dir, "videos"))
        print(
            f"[INFO] Saving video from {camera_keys} for first {args_cli.video_episodes} episode(s) "
            f"at {args_cli.video_fps} fps (stride={args_cli.video_stride}) to: {video_dir}"
        )

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
                for camera_key in camera_keys:
                    _save_camera_snapshot(env, camera_key, snapshot_dir, episode_count, step_count)

            # Open video writers for this episode (if recording is active)
            video_writers = None
            record_this_episode = (
                video_dir is not None and episode_count <= args_cli.video_episodes
            )
            if record_this_episode:
                video_writers = {
                    camera_key: _open_episode_writers(
                        video_dir, camera_key, episode_count, env.num_envs, args_cli.video_fps
                    )
                    for camera_key in camera_keys
                }
                # Record the post-reset frame as the first video frame
                for camera_key, writers in video_writers.items():
                    frames = _grab_camera_frames(env, camera_key)
                    for env_idx, writer in enumerate(writers):
                        writer.append_data(frames[env_idx])

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
                    for camera_key in camera_keys:
                        _save_camera_snapshot(env, camera_key, snapshot_dir, episode_count, step_count)

                # Append a frame to the video writers
                if record_this_episode and step_count % args_cli.video_stride == 0:
                    for camera_key, writers in video_writers.items():
                        frames = _grab_camera_frames(env, camera_key)
                        for env_idx, writer in enumerate(writers):
                            writer.append_data(frames[env_idx])

            # Episode finished — close video writers
            if video_writers is not None:
                for writers in video_writers.values():
                    for writer in writers:
                        writer.close()
                print(f"[INFO] Saved episode {episode_count} video(s) to: {video_dir}")

            sm.print_status(step=step_count, episode=episode_count)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
