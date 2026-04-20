# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hydra entry point for VLA training.

Examples
--------
Behavior cloning on SmolVLA:
    python scripts/train.py mode=bc model=smolvla task=pipetting

RL fine-tune (placeholder — not yet implemented):
    python scripts/train.py mode=rl model=smolvla task=pipetting

Override anything from the CLI:
    python scripts/train.py mode.batch_size=32 mode.optim.lr=5e-5

See configs/train.yaml for all defaults.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Make `import vla` work when running from the repo root without installing.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    mode_name = cfg.mode.name
    if mode_name == "bc":
        from vla.training.bc_train import run_bc
        run_bc(cfg)
    elif mode_name == "rl":
        from vla.training.rl_finetune import run_rl
        run_rl(cfg)
    else:
        raise ValueError(f"Unknown training mode '{mode_name}' (expected 'bc' or 'rl')")


if __name__ == "__main__":
    main()
