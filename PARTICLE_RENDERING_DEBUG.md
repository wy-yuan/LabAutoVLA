# Particle Rendering Debug Notes

## Symptom

In headless Isaac Sim / Isaac Lab runs, PhysX particles were simulated but did
not appear correctly in camera snapshots or rendered output.

The failing workflow was the pipetting workflow with particle fluid enabled:

```bash
isaacpython scripts/run_workflow.py \
  --task LabAuto-Test-Pipetting-Franka-v1 \
  --workflow pipette_liquid \
  --num_envs 1 \
  --enable_cameras \
  --save_snapshots \
  --headless
```

## Things That Did Not Fix It

These extra render/debug flags were tested, but were not required for the final
working path:

```bash
--experience isaaclab.python.headless.rendering.kit
--force_viewport_render
--advance_timeline
--rtx_rendermode RayTracedLighting
--render_physics_steps
```

They can add overhead and make debugging noisier, so keep them off unless a
separate renderer issue specifically needs them.

## Working Fix

The required fix was enabling PhysX-to-USD readback for particle state:

```python
settings = carb.settings.get_settings()
settings.set_bool("/physics/updateToUsd", True)
settings.set_bool("/physics/updateParticlesToUsd", True)
settings.set_bool("/physics/updateVelocitiesToUsd", True)
settings.set_bool("/physics/suppressReadback", False)
```

This keeps the simulated particle positions and velocities authored back into
USD so headless RTX camera rendering can see the updated particle state.

## Where To Set It

Set these after the Isaac Sim application and environment context exist, and
before relying on particle snapshots or camera output. In this repo, the setting
belongs in the MATTErix environment setup path, near particle-system creation.

The current implementation also prints the runtime values:

```text
[INFO] PhysX USD updates: updateToUsd=True, updateParticlesToUsd=True, updateVelocitiesToUsd=True, suppressReadback=False
```

If this line does not appear, or any value is not as expected, particle rendering
may fail in headless snapshots.

## Notes

- Particle-enabled MATTErix configs already disable Fabric with
  `self.sim.use_fabric = False`.
- If Fabric is re-enabled later, it may overwrite USD update settings.
- Keep the minimal run command first; add renderer-specific flags only after
  confirming these PhysX USD update settings are correct.
- For fluid systems, keep the raw PointInstancer hidden. The smooth
  isosurface/material is what gives the liquid its clear light-blue appearance;
  showing particle glyphs can make the liquid look dark or noisy.
