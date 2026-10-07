# Scripted-prefix evaluation

`scripts/evaluate_staged.py` runs the existing dataset-generation state machine
up to a selected boundary, then lets the VLA execute the remaining rollout.
It shares the headless launcher, policy construction, observation adapter, and
normalization loading with `scripts/evaluate.py`.

```bash
isaacpython scripts/evaluate_staged.py task=pipetting handoff.after_stage=picking model.kwargs.pretrained_name_or_path=outputs/YOUR_RUN/ckpt_step030000 model.kwargs.n_action_steps=50 eval.n_episodes=3
```

Keep checkpoint, scene configuration, prompt, and `n_action_steps` identical
when comparing handoff stages. The prompt remains the full task instruction;
no unseen stage-specific prompt is introduced at handoff.

## Stage selection

`handoff.after_stage` selects the **completed** stage:

| Stage | Last primitive index (zero-based) | Model starts from |
| --- | --- | --- |
| `reset` | none | Reset state, with no scripted actions |
| `pre_grasp` | 0 | Pre-grasp pose |
| `open_gripper` | 1 | Open gripper at pre-grasp |
| `grasp` | 2 | Grasp pose, before closing |
| `close_gripper` | 3 | Closed gripper after the configured hold |
| `picking` | 4 | Post-grasp lift completed |
| `source_approach` | 5 | Source approach frame |
| `source_dip` | 6 | Source downward motion completed |
| `source` | 7 | Source lift completed, before moving to target |
| `target_approach` | 8 | Target approach frame |
| `target_dip` | 9 | Target downward motion completed |
| `target` | 10 | Final target lift completed |

`target` is a post-workflow diagnostic; there are no scripted task stages left.
For `task=picking`, named stages are available only through `picking`.
Alternatively, `handoff.after_action=5` hands off after primitive 5 and takes
precedence over `after_stage`. The script logs the indexed primitive sequence.
Named stages reject a changed primitive type sequence to avoid silent misalignment.

```bash
isaacpython scripts/evaluate_staged.py handoff.after_stage=source_approach
isaacpython scripts/evaluate_staged.py handoff.after_stage=source
isaacpython scripts/evaluate_staged.py handoff.after_action=5
```

## Execution and results

- `handoff.max_steps=1200` limits scripted environment steps.
- `eval.max_steps=1200` separately limits model-controlled environment steps.
- The environment's own episode time limit still covers both phases.
- Each episode resets the scene and state machine. At handoff, only model memory
  and the action queue reset; the scene, cameras, and latest observation are kept.
- Script commands go directly to the native environment. Model commands pass
  through the existing VLA action conversion, including gripper scaling.
- Script failure, timeout, or an environment terminal transition before handoff
  skips model execution for that episode. Evaluation proceeds to the next episode.

Outputs go to `outputs/<timestamp>_<model>_eval_staged/`:

- `eval_videos/rollout_epNNN.mp4`: continuous overhead video of both phases,
  recorded at the actual environment control rate.
- `staged_metrics.json`: prefix completion rate, per-episode step counts, stop
  reason, handoff state, and zero-based `handoff_frame`. That frame is the current
  scene just before the first model action. Earlier frames are scripted.
- `.hydra/config.yaml`: resolved run configuration stored by Hydra.

Prefix completion uses the existing workflow criteria, not a new physical grasp
or liquid-transfer check. `handoff_rate` is not task success rate. The report
keeps `terminated` and `truncated` separate and does not interpret all environment
terminations as successful pipetting. No workflow commands are issued after handoff.
