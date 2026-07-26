# Release scope

What this repository contains, what it does not, and why.

---

## Included

### Source
- The complete simulator and RL environment (`src/`): multi-world USD scene
  builder, `DirectMARLEnv` multi-agent environment, observation contract,
  late-fusion actor-critic policy, PhysX and bicycle dynamics backends.
- Training entry point `src/train_student_vehicle_goal_multiagent_rsl_rl.py` and
  evaluation entry point `src/eval_student_vehicle_goal_ppo.py`.
- The weather-to-friction model (`src/trfc/`) with its regression tests.

### Preprocessing
- `scripts/convert_waymo_tfrecord_to_json.py` — WOMD TFRecord → scene JSON.
- `src/scene_factory_config_wizard.py` — interactive scene-pool curation.

### Calibrated vehicle assets
- `artifacts/student_vehicle_assets/vehicle_student/` — the articulated
  10-DOF vehicle as USD + URDF + spec.
- `artifacts/student_vehicle_sysid/comprehensive_fwd_v1_cem_v4/best_config.json`
  — the final CEM-fitted dynamics parameters, loaded automatically at startup.
- `src/student_vehicle_sysid.py` and the teacher-rollout tooling, so the
  calibration can be re-run rather than only re-used.

### Configs
- `configs/scene_factory/` — the demo config and its scene pool.
- `configs/scene_factory/generated/` — the per-experiment configs behind the
  paper's tables, including every eval config.

### Checkpoints
- `checkpoints/v7_weather_aware_iter600.pt` — weather-aware policy.
- `checkpoints/v8_no_weather_iter300.pt` — no-weather baseline.

### Eval scripts
- `run_v8_vs_v7_physics_blind_eval.sh`, `run_v8_vs_v7_moderate_wet_eval.sh`,
  `run_v8_vs_v7_heavy_wet_eval.sh`, `run_bicycle_physx_transfer_eval.sh`,
  `run_v8_physx_to_bicycle_transfer_eval.sh`, `run_scene_diversity_*.sh`,
  `run_eval_pretrained.sh`, plus `scripts/summarize_2x2_eval.py`.

### Documentation
- `README.md` — install and quickstart.
- `docs/configuration.md` — every key in the demo config.
- `docs/friction_model.md` — the friction model and its post-submission correction.
- `scripts/check_install.py` — pre-flight install check.

---

## Not included

### Waymo scene data — **the one thing you must obtain yourself**
The Waymo Open Motion Dataset licence does not permit redistribution, including
of derived scene files. `data/processed/waymo_scenes_json/` ships empty.
**Training and evaluation cannot run until you populate it** — see README §5.
This is the single most common reason a fresh clone fails.

### Benchmark scene splits — *not yet included*
The paper's train/test split (128 train / 64 test WOMD scenarios) is **not
currently in this repository as an explicit ID list.** Scene selection is
reproduced by the `configs/scene_factory/generated/` configs, which name scene
files by index within a converted pool — so an identical split requires an
identical conversion order over the same TFRecord shards. Publishing the WOMD
`scenario_id` list for both splits, which would make the split reproducible
independently of conversion order, is outstanding.

### Isaac Sim and Isaac Lab
Third-party, installed separately (README §2–§3). Isaac Sim is under NVIDIA's
non-commercial research licence.

### Raw sysid teacher datasets
The recorded PhysX teacher rollouts behind the CEM fit are not shipped (size).
The scripts that regenerate them are.

### Training logs
`logs/` is gitignored. Only final checkpoints are shipped.

---

## Known limitations

These are properties of the current implementation, not bugs:

- **Road geometry is flattened to z = 0** (`flatten_road_z: true` by default). No
  bridges, overpasses, ramps, banking or elevation.
- **Road surfaces are decorative.** `enable_segment_collision=False` in
  `src/chocolate_waymo_builder.py`; vehicles drive on a flat ground plane, and
  lane boundaries are enforced by reward penalties, not physics.
- **Road observations are a privileged static map**, not perception — brute-force
  k-NN against polyline points loaded at scene-build time. There is no sensor
  simulation and no perception noise, so results are not end-to-end autonomous
  driving performance.
- **Friction is uniform per world.** One μ per world, applied to that world's
  vehicles' wheel materials. No puddles, lane-dependent surfaces, or
  within-episode weather change.
- **Static obstacles and parked vehicles are removed** during conversion.
- **Scene cropping is silent.** `chocolate_waymo_builder.py` drops polylines
  outside `bounds_size_m` or with gaps longer than `jump_break_m`, so a scene's
  JSON can contain roads the environment never sees.
- **The vehicle is calibrated against Isaac Sim's built-in PhysX Vehicle model,
  not against measured real-vehicle data.** The platform is best described as
  configurable and physically motivated rather than validated against real
  vehicles.
- **The friction model is validated against the published curves of the cited
  paper, not against measured tyre–pavement friction.** See
  `docs/friction_model.md`, including the one acceptance criterion that does not
  pass because the source paper is internally inconsistent.
- **Single GPU only.** No multi-GPU or multi-node support.

---

## Post-submission changes

Changes made to this repository after the paper was submitted:

| change | effect |
|---|---|
| Eq. (12) coefficient `A` corrected from a contact-patch area to a dimensionless constant (4.05); Stribeck exponent α set to 0.90 | μ no longer collapses to exactly 0 above a 0.8 mm water film. **Affects any wet-condition result.** See `docs/friction_model.md`. |
| `requirements.txt` added | `stable-baselines3` was missing from the documented install, which broke every eval script at import. |
| Isaac Lab install instruction corrected | the README instructed `git checkout v0.54.3`, which is a package version, not a git tag, and does not exist. |
| `IsaacLab/` submodule gitlink removed | a fresh clone got an empty directory with no `.gitmodules`. |
| `scripts/check_install.py` added | install problems now report in ~1 s instead of after an Isaac Sim boot. |
| `docs/configuration.md`, `docs/friction_model.md`, `RELEASE.md` added; README rewritten | |
