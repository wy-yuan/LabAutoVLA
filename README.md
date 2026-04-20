# LabAutoVLA
- GPU-accelerated simulation environment using NVIDIA Isaac Sim and Isaac Lab framework for realistic lab automation tasks 
- VLA fine-tuning framework (SmolVLA with behavior cloning) integrated with simulation assets

## Simulation Demo

<img src="./demo/episode_001_env_0.gif" width="224" alt="pipetting demo" />

## Install
1. Install Isaacsim and Isaaclab
```bash
cd C:\isaacsim5.1
post_install.bat
C:\isaacsim5.1\python -m pip install -e isaaclab_path\source\isaaclab
C:\isaacsim5.1\python -m pip install -e isaaclab_path\source\isaaclab_tasks
# note the version of these packages: usd-core==25.5, lxml==4.9.2, h5py==3.15.1
```
2. Install matterix source in editable mode
```bash
C:\isaacsim5.1\python -m pip install -e .\source\*
```

3. Install other requirements
```bash
C:\isaacsim5.1\python -m pip install -r requirements.txt --upgrade-strategy only-if-needed # Avoid conflicts
# Cuda torch should work after Isaaclab installation. If it fails, reinstall with: 
C:\isaacsim5.1\python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

4. Set PATH
```bash
$env:MATTERIX_PATH = "[your_project_path]\LabAutoVLA"
```

## Modeling

### Dataset Generation (Synthetic Data)

Generate high-quality training datasets directly from simulation with automatic language variation and domain randomization:

#### Single Task (Pipetting, 1000 episodes):
```bash
C:\isaacsim5.1\python -m data.generate_dataset stages.collect_hdf5=true stages.convert_lerobot=true
```

**Parquet columns:**
- `observation.state`: (D_state,) float32 — proprioceptive state (EE pos + quat + gripper)
- `observation.images.overhead`: video — camera frames (referenced in MP4)
- `action`: (8,) float32 — Franka IK action (EE pos, rot, gripper)
- `task`: str — language instruction (different for every episode)
- `task_type`: str — task ID (for multi-task datasets)

#### Training on Generated Data
```bash
C:\isaacsim5.1\python scripts/train.py mode=bc model=smolvla task=pipetting mode.sim_eval=false mode.epochs=1 mode.batch_size=2 mode.num_workers=0
```
---

### Record Manual Demos

1. Record demos with existing workflow:
   ```bash
   C:\isaacsim5.1\python scripts\run_workflow.py --task LabAuto-Test-Pipetting-Franka-v1 --workflow pipette_liquid --num_envs 1 --enable_cameras --save_video --headless --livestream=2 # or --livestream=1
   ```

---