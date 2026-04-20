"""RL fine-tuning placeholder.

Not wired to a working algorithm yet. This file documents the intended
integration points so a future contributor can drop in PPO/DPPO/RLFT
without restructuring the codebase.

Recommended approach — order of implementation
----------------------------------------------

1. **Warm-start from a BC checkpoint**
   Load the VLA with ``vla.load_pretrained(cfg.mode.bc_ckpt)``. We never
   train a VLA from scratch with RL — sample efficiency is terrible.

2. **Wrap the Matterix env as a vectorised RL env**
   Matterix already runs N parallel envs on the GPU — just plumb through
   a reward function. Put task-specific reward code in
   ``source/matterix_tasks/.../rewards.py`` so multiple algorithms can
   reuse it.

   The ``VLAEnvWrapper`` already handles the observation adaptation; add
   a small ``RewardWrapper`` on top for reward shaping / normalisation.

3. **Policy interface**
   ``BaseVLA.predict_action`` returns deterministic actions. For RL you
   need a stochastic head:
       * Cheap path: Gaussian on top of the existing action prediction
         — add ``predict_action_dist(images, state, task) -> Normal``
         as a **new method on BaseVLA** (optional, default to None), and
         fail loudly here if the chosen VLA doesn't implement it.
       * Proper path: reuse the flow-matching head and apply score-based
         RL (e.g. DPPO / FlowRL). SmolVLA's backbone already produces a
         velocity field — wire its samples into the RL objective.

4. **Algorithm choice**
   * PPO (via ``stable-baselines3`` or ``torchrl``) — simplest, stable.
   * DPPO — diffusion/flow-matching PPO variant; keeps SmolVLA's action
     distribution intact.
   * AWR / IQL — fully offline; can be warm-started from the same
     LeRobotDataset used for BC.

5. **KL-to-BC regulariser**
   Always keep a KL term against the frozen BC-init policy; prevents
   the VLA from catastrophically forgetting the imitation prior.

Extension checklist
-------------------
When you implement this file for real:
    [ ] Add ``cfg.mode.bc_ckpt`` to configs/mode/rl.yaml (path to BC ckpt)
    [ ] Add ``cfg.mode.algo`` (ppo | dppo | iql | awr)
    [ ] Add ``cfg.task.reward`` (name of reward term in Matterix)
    [ ] Add a ``predict_action_dist`` method to any VLA you want to RL-tune
    [ ] Share ``evaluator.py`` for eval — no need to duplicate
"""

from __future__ import annotations

import logging

from omegaconf import DictConfig

log = logging.getLogger(__name__)


def run_rl(cfg: DictConfig) -> None:
    raise NotImplementedError(
        "RL fine-tuning is a scaffolded placeholder. See docstring in "
        "vla/training/rl_finetune.py for the recommended implementation "
        "plan. Start by BC-training with `mode=bc`, then come back here."
    )
