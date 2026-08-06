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
- The extension points, for building on the simulator without editing it:
  `src/scene_factory_scenario_api.py` (point obstacles, oriented keep-out
  regions, per-zone speed limits, placed or swapped at runtime) and
  `src/scene_factory_od_registry.py` (`register_od_mode(name, fn)`, with the
  registering package named per config). Obstacles live in two places — the
  collision registry and the road-point pool the policy perceives — and the
  scenario API is what keeps them in sync.
- Validation harnesses: `src/brake_friction_sweep.py` (NHTSA-referenced braking,
  driven by `run_brake_sweep_replicates.sh`) and `src/traction_probe.py`.
- `SF_FORMAT.md` — the versioned scene schema (`sf_version: "2.0"`), plus
  `scripts/migrate_scene_schema.py` for older scenes.

### Preprocessing
- `scripts/convert_waymo_tfrecord_to_json.py` — WOMD TFRecord → scene JSON.
- `src/scene_factory_config_wizard.py` — interactive scene-pool curation.

### Calibrated vehicle assets
- `artifacts/student_vehicle_assets/vehicle_student/` — the articulated
  10-DOF vehicle as USD + URDF + spec.
- `artifacts/student_vehicle_sysid/comprehensive_fwd_v1_cem_v4/best_config.json`
  — the CEM-fitted dynamics parameters, loaded automatically at startup.
- `src/student_vehicle_sysid.py` and the teacher-rollout tooling, so the
  calibration can be re-run rather than only re-used.

The suspension spring and damper are set at run time rather than read from that
file. `physx_suspension_stiffness_n_m` (2.0 MN/m) and
`physx_suspension_damping_ns_m` (49 kN·s/m, ≈79% of critical) are config fields,
and they are the operative values. The CEM objective is open-loop trajectory
match and does not constrain spring rate, so if you re-run the sysid, keep the
runtime override: a spring soft enough to let the 1800 kg chassis rest on its
travel stops delivers normal force as constraint spikes rather than as a spring
force, and nothing downstream of tire load will be meaningful.

### Configs
- `configs/scene_factory/` — every config the documented commands use: the demo,
  the two weather-exposure training arms, the bicycle arm, braking validation,
  and the multi-seed eval configs, each alongside its scene pool.

- `configs/scene_factory/generated/` — the **scene pools**, and only those: the
  256-scene training pools for each weather arm, the held-out 199-scene eval pool
  in its four surface conditions, and the scene-diversity ablation pools. Each
  lists its scene files by name, so these are the splits themselves rather than a
  description of them. The rest of that directory is machine-generated scratch
  written against pools that no longer exist, and is not shipped.

### Checkpoints
- `checkpoints/weather_exposed_iter900.pt` — trained on 0–12 mm water films.
- `checkpoints/dry_trained_iter900.pt` — same architecture and scene pool,
  trained all-dry.

These are the two arms of the weather-exposure ablation and the policies the
friction-conditioning and transfer scripts load by default, so those run without
training anything. `run_eval_pretrained.sh` evaluates both in about 20 minutes
and is the fastest end-to-end check of an install.

A bicycle-backend policy is **not** shipped — `run_transfer_eval.sh` needs one
and will tell you to train it with `run_bicycle_dry_rebuttal.sh`. It must be
speed-matched to the PhysX vehicle (`bicycle_max_speed_mps: 4.5`); an unmatched
bicycle policy is solving a different problem and its transfer number does not
mean what it appears to.

### Validation and eval scripts
- `run_brake_sweep_replicates.sh` — NHTSA-referenced braking, ten seeded sweeps,
  aggregated by `scripts/plot_brake_sweep.py --replicates <dir>`. Uses no policy
  and no scene data, so it runs immediately after install; ~4 minutes total.
- `run_physics_sanity_test.sh` — bit-for-bit deterministic, no scene data. Any
  change in its numbers is a real behavioural change, never noise, which makes it
  the way to prove a refactor changed nothing.
- `run_traction_probe.sh` — distinguishes driving from free fall. Worth having:
  a vehicle whose wheels carry no normal force still reports a trajectory.
- `run_weather_ablation_rebuttal.sh`, `run_bicycle_dry_rebuttal.sh` — the two
  training arms of the weather-exposure ablation, plus its bicycle counterpart.
- `run_paper_table4_eval.sh`, `run_transfer_eval.sh` — friction conditioning and
  the cross-dynamics transfer matrix, multi-seed, aggregated by
  `scripts/aggregate_table4_eval.py` and `scripts/aggregate_transfer_eval.py`.
- `run_scene_diversity_*.sh`, `run_eval_pretrained.sh`.

### Documentation
- `README.md` — install and quickstart.
- `docs/configuration.md` — every key in the demo config.
- `docs/friction_model.md` — the friction model, its two calibrated parameters,
  and how to re-derive them.
- `scripts/check_install.py` — pre-flight install check.

---

## Not included

### Waymo scene data — **the one thing you must obtain yourself**
The Waymo Open Motion Dataset licence does not permit redistribution, including
of derived scene files. `data/processed/waymo_scenes_json/` ships empty.
**Training and evaluation cannot run until you populate it** — see README §5.
This is the single most common reason a fresh clone fails.

### Benchmark scene splits — *partially included*
The split is 455 usable scenes, 256 train / 199 eval, disjoint. The scene-pool
YAMLs that define it **are** shipped, so the split is reproducible — but they
name scene files by their converted filename (`scene_000001.json`), not by WOMD
`scenario_id`, so reproducing it requires an identical conversion order over the
same TFRecord shards. Publishing the `scenario_id` list, which would make the
split reproducible independently of conversion order, is outstanding.

Coverage is uneven and worth knowing before you read a result as general: 59.6%
urban grid, 32.5% multi-junction, no roundabouts and effectively no merges.
Attrition from the raw shards is mostly parked vehicles (62.9%) and
out-of-crop scenes (6.2%).

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
- **Three different things get called "validation" here, and they are not
  interchangeable.** *Calibration*: the vehicle is CEM-fitted to Isaac Sim's
  built-in PhysX Vehicle model, not to a real car. *Implementation check*: the
  friction model reproduces the published curves of the cited paper, not
  measured tyre–pavement friction — see `docs/friction_model.md`, including the
  one acceptance criterion that does not pass because the source paper is
  internally inconsistent. *External reference*: braking distance is compared
  against NHTSA measured data (`run_brake_sweep_replicates.sh`), which is the only
  place the platform is checked against something outside its own stack. Treat
  it as configurable and physically motivated, except on braking.
- **Single GPU only.** No multi-GPU or multi-node support.

---

## Before you trust a number

Four things fail quietly rather than loudly. Each costs minutes to check and can
invalidate a whole run if you don't.

**Does your μ range actually bind?** The vehicle tops out around 5 m/s, so
acceleration is friction-limited only below μ ≈ 0.18 and cornering below μ ≈ 0.36.
Above that, friction is present but never the binding constraint, and a weather
ablation run there cannot detect what it is designed to detect — it will return a
clean null. Emergency braking is the exception; it touches the envelope at any μ.
Check the range before spending GPU-days on it.

**Are your ground slabs overlapping?** Each world gets its own 1000 m kinematic
ground cuboid. Keep `env_spacing > ground_cuboid_size_m`. Overlapping slabs
change contact behaviour, and the `contact_offset` default was tuned on the
non-overlapping case.

**Are your agents actually driving?** `bash run_physics_sanity_test.sh` needs a
GPU but no scene data, and is bit-for-bit deterministic — so any change in its
numbers is a real behavioural change, never noise. `run_traction_probe.sh`
distinguishes driving from free fall, which is worth having: a vehicle whose
wheels carry no normal force still reports a trajectory.

**Are worlds coupled?** They share one PhysX scene and couple through
solver-level reductions even at large `env_spacing`. Cross-map batched numbers
are not trustworthy — run one map per process.

The CPU-only regression tests need no GPU and no scene data:

```bash
PYTHONPATH=. python -m pytest src/trfc/tests -q
```
