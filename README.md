# LabAutoVLA
- GPU-accelerated simulation environment using NVIDIA Isaac Sim and Isaac Lab framework for realistic lab automation tasks 
- VLA fine-tuning framework (SmolVLA with behavior cloning) integrated with simulation assets

## Install
1. Install Isaacsim and Isaaclab

2. Install matterix source in editable mode
``` C:\isaacsim\python -m pip install -e .\source\* ```
3. Install other requirements
```pip install -r requirements.txt```

## Simulation Demo

<video src="./demo/episode_001_env_0.mp4" controls width="224"></video>

> If the embedded video does not render on your Git host, open [demo/episode_001_env_0.mp4](./demo/episode_001_env_0.mp4) directly.

## Modeling
### 1. Record demos with existing workflow:
C:\isaacsim\python scripts/run_workflow.py --task LabAuto-Test-Pipetting-Franka-v1 --workflow pipette_liquid --num_envs 4
### 2. Train (auto-converts HDF5 -> LeRobot on first run):
C:\isaacsim\python scripts/train.py mode=bc model=smolvla task=pipetting
