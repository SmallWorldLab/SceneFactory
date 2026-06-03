# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

SceneFactory is a GPU-vectorized multi-agent RL platform for autonomous driving simulation. It runs on NVIDIA Isaac Sim 5.1.0 + Isaac Lab 0.54.3, trains policies with RSL-RL (PPO), and represents worlds and agents as batched GPU tensors. The main use case: train a vehicle to reach goals across a pool of real Waymo road scenes, with optional weather/friction conditioning.

The `workzone` branch (current) extends this to construction-zone scenarios: synthetic straight-road scenes with randomized taper layouts, a bicycle-dynamics training mode, and a CEM optimizer for taper design.

## Running Things

All scripts must be run from the repo root with `PYTHONPATH=.`. The conda env name in most scripts is resolved dynamically; `run_workzone_demo.sh` uses `isaac-pytorch` explicitly — this may differ from the `scenefactory` name in the README.

```bash
# Demo training (recommended starting point — full feature stack, ~8–10 GB VRAM)
bash run_demo_train.sh

# Physics sanity test (no Waymo data needed — straight-line throttle test)
bash run_physics_sanity_test.sh

# Visualize scenes in Isaac Sim GUI
bash run_visualize_scene.sh --world_count 4

# Workzone taper demo (synthetic scenes, GUI)
bash run_workzone_demo.sh [--num_worlds 9] [--headless]

# Workzone training
bash run_workzone_train.sh [--no-headless]

# Evaluate pre-trained checkpoints (paper results)
bash run_eval_pretrained.sh

# Evaluate a trained run
PYTHONPATH=. python src/eval_student_vehicle_goal_ppo.py \
  --run_dir logs/rsl_rl/<experiment>/<run> --headless

# Convert Waymo TFRecords → scene JSONs (requires Python 3.10 + waymo-open-dataset)
python scripts/convert_waymo_tfrecord_to_json.py \
  --tfrecord-dir data/waymo_tfrecords \
  --output-dir data/processed/waymo_scenes_json

# Generate workzone training scenes
python -m src.workzone_scene_generator \
  --output_dir data/processed/workzone_scenes_json \
  --num_worlds 32 --seed 42

# Friction model CLI (no Isaac Sim needed)
python -m src.trfc --road-type AC --water-film-mm 0.2
python -m src.trfc --demo
```

Training entry point accepts `--config`, `--headless`, `--num_envs`, `--num_agents_per_env`, and many per-field config overrides. Config values can also be overridden via CLI flags matching YAML keys (e.g., `--ground_mode cuboid`).

## Isaac Lab Location

`src/isaaclab_bootstrap.py:ensure_isaaclab_source_paths()` adds `../../IsaacLab/source/{isaaclab,isaaclab_rl,isaaclab_assets}` to `sys.path` — that is, a **sibling of this repo** at `../IsaacLab/`. The `IsaacLab/` directory inside this repo is an empty placeholder (noted in WORKZONE.md as OSM-01: not yet a proper git submodule). Every source file calls `ensure_isaaclab_source_paths()` before importing `isaaclab.*`.

## Architecture

### Data Flow

```
Waymo TFRecords  ──→  scripts/convert_waymo_tfrecord_to_json.py
                           │
                           ▼
                   data/processed/waymo_scenes_json/*.json   (scene pool)
                           │
                   src/trfc/world_pipeline.py  (StageWorldSpec per world)
                    ├── assigns scene JSON to each world index
                    └── computes friction estimate (ALL model) per world
                           │
                   src/chocolate_waymo_builder.py  (_build_road_world_from_json)
                    ├── reads polylines, builds USD cuboid strips per road segment
                    ├── stores road points as USD custom metadata on the world prim
                    └── enables_segment_collision=False  ← road is DECORATIVE only
                           │
                   Isaac Sim USD stage (one Xform per world, grid layout)
```

Workzone training instead uses `src/workzone_scene_generator.py` to produce synthetic JSONs with the same schema, then routes them through the same builder.

### RL Environment

`src/student_vehicle_multiagent_goal_env.py` — main class, extends `DirectMARLEnv`. Key internals:

- **Lane touch metadata**: at scene load, road polyline XYZ points are read from USD custom metadata into `_lane_touch_points_xy_m [num_envs × N_points × 2]` (GPU tensor). All road observation and reward queries are brute-force k-NN distance lookups against this static tensor — **this is a privileged map, not perception**. Closing a workzone lane does not update it (see WORKZONE.md P0-08).
- **Ground**: vehicles drive on a per-world 1000×1000×1m kinematic cuboid (`ground_mode="cuboid"`) or a shared infinite plane (`ground_mode="plane"`). Road surface collision is disabled.
- **Dynamics modes**: `physx` (articulated 10-DOF PhysX vehicle, sysid-calibrated) or `bicycle` (kinematic bicycle model, same obs/action interface, used for workzone training because PhysX has a torque plateau issue at low speeds).

### Observation

`src/scene_factory_obs_contract.py` defines the vector structure:

| Segment | Dim | Content |
|---|---|---|
| ego | 7 | velocity, heading, goal direction, etc. |
| weather | 4 | `[h_w/1.0, 1_AC, 1_SMA, 1_OGFC]` |
| road points | k × 3–5 | nearest k road center points (XY + type + optional dirs) |
| neighbor vehicles | k × 6–7 | relative position/velocity/heading per neighbor |

`src/trfc/observation_context.py:encode_weather_context()` produces the 4D weather token from water-film depth (mm) + one-hot road surface type.

### Policy

`src/scene_factory_late_fusion_actor_critic.py` — `_StructuredLateFusionBackbone`: separate MLP branches for ego/road/vehicle, each with LayerNorm + dropout, fused via max-pool over the set encodings, then a shared trunk. Registered as a custom RSL-RL actor-critic class.

### Friction Model

`src/trfc/friction_api.py` implements the modified ALL (Average Lumped LuGre) model from Zhao et al. (2024). `estimate_friction(FrictionInput) → FrictionEstimate` returns scalar μ. `src/trfc/world_pipeline.py:prepare_stage_world_specs()` calls this per world and the result is applied as a global PhysX friction scalar on the ground cuboid — **not per surface patch** (per-zone friction is a future item WZ-08).

### Config System

Single YAML hierarchy controls everything. Training configs live in `configs/scene_factory/` and have sections: `env`, `scene_factory`, `observation`, `reward`, `policy`, `assets`, `runner`, `app`. The `scene_factory.config_path` field points to a separate scene-pool YAML that lists scene JSON files and per-world weather assignments. `src/scene_factory_config_wizard.py` is an interactive Isaac Sim tool for curating scene pools visually.

Generated/experiment configs land in `configs/scene_factory/generated/`.

### Vehicle System Identification

`src/student_vehicle_sysid.py` — CEM optimizer fitting a `StudentTunableConfig` (bicycle-model parameters) to match teacher PhysX rollouts. Pre-fitted params at `artifacts/student_vehicle_sysid/comprehensive_fwd_v1_cem_v4/best_config.json` are loaded at env startup via `load_tunable_config()` and applied by `_apply_runtime_student_dynamics()`.

## Scene Format

Scene JSONs follow a Waymo-derived schema: `meta`, `road.polylines` (each with `type`, `id`, `n`, `xyz`, `dir`), `agents`. Polyline type codes: 1=lane_center (drivable), 2=lane_boundary_left, 3=lane_boundary_right, 4=road_edge, 15=road_edge (used in synthetic workzone scenes), 6=broken divider. The SF Format spec (`SF_FORMAT.md`) defines a versioned upgrade to this schema (adds `sf_version`, `terrain`, `surfaces`, `workzone`, `passable` per polyline) but migration is not yet complete.

## Known Architectural Constraints

- **Road collision is disabled**: `enable_segment_collision=False` in `chocolate_waymo_builder.py`. Lane boundaries are enforced only via reward penalty terms, not physics.
- **Waymo Z is discarded** by default (`flatten_road_z: true`). The ground plane is flat regardless of real road elevation.
- **Friction is per-world, not per-surface**: a single scalar applied to the whole ground cuboid per world. Workzone surface patches (gravel, steel plate) have no physical friction effect yet.
- **Road map is static**: closing a lane (workzone) does not update `_lane_touch_points_xy_m` — the policy's map still sees the lane as open.

These are tracked as Phase 0 items in `WORKZONE.md`.

## Logs and Artifacts

- Training checkpoints/logs: `logs/rsl_rl/<experiment_name>/<run_name>/`
- Pre-trained checkpoints: `checkpoints/v7_weather_aware_iter600.pt`, `checkpoints/v8_no_weather_iter300.pt`
- Vehicle USD + sysid params: `artifacts/student_vehicle_assets/`, `artifacts/student_vehicle_sysid/`
- Workzone demo output: `artifacts/scene_factory/workzone_demo/`
