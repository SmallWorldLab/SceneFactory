# SceneFactory

**SceneFactory** is a GPU-vectorized platform for procedural scene construction, physics-based multi-agent simulation, and reinforcement learning in autonomous driving environments.

Built on [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) and [Isaac Lab](https://github.com/isaac-sim/IsaacLab), SceneFactory represents worlds and agents as batched tensors — vehicle control, observations, rewards, resets, and policy inference are all GPU tensor operations.

> Paper: *SceneFactory: GPU-Accelerated Multi-Agent Driving Simulation with Physics-Based Vehicle Dynamics* (under review)

---

## Read this first

**You cannot run training or evaluation from a fresh clone alone.** SceneFactory
drives on road geometry derived from the Waymo Open Motion Dataset. WOMD is not
redistributable, so this repository ships **no scene data** — you must download
WOMD yourself and run the conversion script. That step is
[§5](#5-prepare-scene-data-required) and it is not optional.

Budget roughly:

| step | time | disk |
|---|---|---|
| Isaac Sim + Isaac Lab install | 30–60 min | ~14 GB |
| WOMD download (1 shard is enough to start) | 10–30 min | ~1 GB+ |
| Scene conversion | ~1 min per shard | small |

At any point, run the pre-flight check to see exactly what is missing:

```bash
PYTHONPATH=. python scripts/check_install.py
```

It validates the interpreter, every required package, the Isaac Lab layout, the
vehicle assets, the friction model, and the scene files your config references —
in about a second, without starting Isaac Sim.

---

## Key Features

- **8,192 worlds on one GPU** — peak **34,834 CASPS** (controlled-agent simulation steps per second). Against single-embodiment benchmarks the comparable unit is **31,527 env-steps/s**, also at 8,192 envs. Capacity is set by the observation vector (6.0 MB/world at `d_obs = 387`), so `road_points_k` is the knob that moves it
- **42× faster per step** than a non-batched PhysX vehicle at a fixed world count (5,683 → 135 ms). Compounded end to end the gain reaches 127×. Both are measured on fixed hardware inside a physics-based pipeline — they say what a batchable vehicle buys you there, not how this compares to a kinematic simulator
- **Tensor-resident articulated vehicle** — a 10-DOF PhysX articulation whose joint states, actuation and contact materials all live in the batched pipeline, calibrated by system identification. This is the piece that makes the rest batchable: Isaac Lab ships no drivable car, and the PhysX Vehicle SDK keeps wheel, tire and drivetrain state outside the tensor pipeline, which forces per-agent traversal
- **Waymo Open Motion Dataset** scenes converted to USD road environments offline — **455 usable scenes, 256 train / 199 eval, disjoint**. Topology coverage is uneven by construction: 59.6% urban grid, 32.5% multi-junction, no roundabouts and effectively no merges
- **Weather-to-friction module** implementing the modified ALL (Average Lumped LuGre) model — maps precipitation and road surface to a PhysX friction coefficient per world, each world carrying its own ground material
- **Externally referenced braking** — the vehicle's 100 km/h no-ABS stopping distance is checked against NHTSA VRTC-87-0441 measured data, landing at 57 m dry and 69 m wet, inside the measured bands (`bash run_brake_sweep_replicates.sh`)
- **Runtime scenario API** — point obstacles, oriented keep-out regions and per-zone speed limits, placed or swapped at runtime, plus a pluggable origin–destination mode registry (`register_od_mode`) declared per config
- **Bicycle-model backend** (`dynamics_mode: bicycle`) for fast prototyping and transfer studies, sharing the same observation/action/reward interface as PhysX
- **Single YAML hierarchy** controls everything: scene pool, parallelism, weather distribution, dynamics backend, and all reward weights. Scenes are versioned (`sf_version: "2.0"`, see [SF_FORMAT.md](SF_FORMAT.md))

---

## Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| OS | Linux | Isaac Sim 5.1.0, tested on Ubuntu 22.04/24.04 |
| Python | **3.11** | Isaac Sim 5.1.0 ships `cp311` wheels only |
| NVIDIA Isaac Sim | 5.1.0 | installed via pip, see §2 |
| Isaac Lab | package `0.54.3` = commit `e10c302cd2` | see §3 — **the version numbering is confusing, read it** |
| RSL-RL | 3.1.2 | `rsl-rl-lib` |
| PyTorch | 2.7.0 | pulled in by Isaac Sim |
| NVIDIA driver | 580.173 verified | consult the [Isaac Sim 5.1.0 requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html) for the supported minimum |

### GPU / VRAM

| configuration | VRAM | source |
|---|---|---|
| paper experiments — 256 worlds × 16 agents, `road_points_k: 350` | ~90 GB | measured, RTX PRO 6000 (96 GB) |
| `run_demo_train.sh` — 32 worlds × 4 agents, `road_points_k: 64` | ~8–10 GB | measured |

**We have not measured SceneFactory on 16 GB or 24 GB consumer GPUs.** Reported
per-env memory is ≈23 MB, so the demo's 128 agent slots should fit in 24 GB and
probably in 16 GB, but we are not stating a number we have not run. If you are on
a smaller card, scale down and report what you see:

```bash
bash run_demo_train.sh --num_envs 16 --num_agents_per_env 2 --obs_road_points_k 32
```

`env.num_envs`, `env.num_agents_per_env` and `observation.road_points_k` are the
three knobs that dominate memory; `road_points_k` is usually the cheapest to cut
(CLI: `--obs_road_points_k`).

---

## Installation

Isaac Sim, Isaac Lab and SceneFactory must end up in a specific **directory
layout**. `src/isaaclab_bootstrap.py` resolves Isaac Lab as a *sibling* of this
repository:

```
your-workspace/
├── IsaacLab/          ← cloned in §3
└── SceneFactory/      ← this repo
```

Cloning IsaacLab *inside* `SceneFactory/` will not work.

### 1. Create the environment

```bash
mkdir -p ~/your-workspace && cd ~/your-workspace
conda create -n scenefactory python=3.11 -y
conda activate scenefactory
```

### 2. Install Isaac Sim

```bash
pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com
pip install isaacsim-rl==5.1.0 --extra-index-url https://pypi.nvidia.com
```

This downloads ~14 GB of extension caches. Subsequent runs reuse them.

### 3. Install Isaac Lab from source

> **The two version numbers.** Isaac Lab has a repository git tag (`v2.3.2`, …)
> *and* a separate `isaaclab` Python package version (`0.54.3`), declared in
> `source/isaaclab/config/extension.toml`. The paper cites the **package**
> version, 0.54.3. **`0.54.3` is not a git tag** — `git checkout v0.54.3` fails
> with `pathspec 'v0.54.3' did not match any file(s) known to git`. Earlier
> versions of this README told you to run exactly that. It was wrong.
>
> There is also no tag that *equals* package 0.54.3: tag `v2.3.2` is package
> 0.54.2, and 0.54.3 lands six commits later on `main`. So we pin the commit.

```bash
cd ~/your-workspace
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout e10c302cd247eb98201067952fcd38979074fe3d   # isaaclab package 0.54.3
pip install -e source/isaaclab
pip install -e source/isaaclab_assets
pip install -e source/isaaclab_tasks
cd ..
```

Verify:

```bash
python -c "import importlib.metadata as m; print(m.version('isaaclab'))"   # 0.54.3
```

If you would rather track a release tag, `git checkout v2.3.2` gives package
0.54.2, which is one patch behind what the paper used. We have not validated
SceneFactory against it.

### 4. Clone this repo and install the remaining dependencies

```bash
cd ~/your-workspace
git clone https://github.com/SmallWorldLab/SceneFactory.git
cd SceneFactory
pip install -r requirements.txt
```

`requirements.txt` covers only the plain-PyPI dependencies — Isaac Sim and Isaac
Lab are installed in §2 and §3 and deliberately are not listed there.

Check the install:

```bash
PYTHONPATH=. python scripts/check_install.py
```

Everything except the scene-data check should pass. If `IsaacLab source tree`
fails, your directory layout does not match the diagram above.

### 5. Prepare scene data (required)

Download the [Waymo Open Motion Dataset](https://waymo.com/open/data/motion/)
(accepting the Waymo licence is required) and place one or more `*.tfrecord`
shards in `data/waymo_tfrecords/`. The `data/` tree is not in the repository —
nothing under it is redistributable — so create it first:

```bash
mkdir -p data/waymo_tfrecords data/processed/waymo_scenes_json
```

> **Get the right variant.** The converter reads the **`tf_example`** format —
> it parses `roadgraph_samples/{xyz,dir,id,type,valid}` features. In the WOMD
> bucket these are under `uncompressed/tf_example/training/`, with filenames like
> `uncompressed_tf_example_training_training_tfexample.tfrecord-00000-of-01000`
> (~1.2 GB each). The **`scenario`** protos under `uncompressed/scenario/` are a
> different serialization and **will not parse** with this script. One shard is
> plenty to start.

The converter depends on `waymo-open-dataset-tf-2-12-0`, which
[does not support Python 3.11](https://github.com/waymo-research/waymo-open-dataset/issues/868).
Run it in a **separate Python 3.10 environment** — this env is only used for the
conversion and is never used to run the simulator:

```bash
conda create -n waymo-extract python=3.10 -y
conda activate waymo-extract
pip install tensorflow==2.12.0 waymo-open-dataset-tf-2-12-0 numpy

python scripts/convert_waymo_tfrecord_to_json.py \
  --tfrecord-dir data/waymo_tfrecords \
  --output-dir data/processed/waymo_scenes_json

conda activate scenefactory   # switch back
```

This writes `scene_000000.json`, `scene_000001.json`, … to
`data/processed/waymo_scenes_json/`, which is the default `io.scene_json_dir` in
every shipped config.

`run_demo_train.sh` needs the eight scenes named in
`configs/scene_factory/demo_weather_scenes.yaml` (`scene_000000`, `000002`,
`000003`, `000006`, `000011`, `000014`, `000016`, `000017`). One WOMD shard
produces far more than that. Confirm with:

```bash
PYTHONPATH=. python scripts/check_install.py
```

---

## Quickstart

### Demo training

Exercises the full feature stack — sysid-calibrated PhysX vehicle,
weather-to-friction module, real Waymo road geometry — at single-GPU scale:

```bash
bash run_demo_train.sh
```

**32 worlds × 4 agents** across four weather conditions (dry / light / moderate /
heavy rain), 500 iterations, ~30–60 min on an RTX 3090 / A100, ~8–10 GB VRAM.
Logs and checkpoints go to `logs/rsl_rl/scene_factory_demo/demo_weather_physx/`.

To ablate the weather module, add `--obs_weather_context_blind`, or set
`observation.weather_context_blind: true` in
`configs/scene_factory/demo_weather_physx_train.yaml`.

(Note the `obs_` prefix, and that it is a `BooleanOptionalAction` flag — pass it
bare, or `--no-obs_weather_context_blind` to force it off. `--obs_weather_context_blind true`
is a parse error.)

### Evaluate pre-trained policies

| File | Description |
|---|---|
| `checkpoints/weather_exposed_iter900.pt` | Trained with 0–12 mm water films; conditions on the weather token |
| `checkpoints/dry_trained_iter900.pt` | Same architecture and scene pool, trained all-dry — the baseline |

```bash
bash run_eval_pretrained.sh
```

~5 min per condition; results are written to timestamped directories.

### Visualize scenes

```bash
bash run_visualize_scene.sh --world_count 4
```

### Other training entry points

```bash
# PhysX, all-dry — the dry arm of the weather-exposure ablation
PYTHONPATH=. python src/train_student_vehicle_goal_multiagent_rsl_rl.py \
  --config configs/scene_factory/waymo_physx_256_train_16agents_knn_alldry.yaml \
  --headless

# friction-aware — same config but for the scene pool's water-film field
PYTHONPATH=. python src/train_student_vehicle_goal_multiagent_rsl_rl.py \
  --config configs/scene_factory/waymo_physx_256_train_16agents_knn_wet0to12.yaml \
  --headless

# bicycle backend, speed-matched to the PhysX vehicle, for transfer ablation
bash run_bicycle_train.sh --headless
```

The two arms differ in exactly two substantive lines — scene pool and run name —
and the pools themselves differ in one field, `water_film_mm`. That is the whole
ablation, and it is worth checking rather than trusting:

```bash
diff <(grep -v '^\s*#' configs/scene_factory/waymo_physx_256_train_16agents_knn_alldry.yaml) \
     <(grep -v '^\s*#' configs/scene_factory/waymo_physx_256_train_16agents_knn_wet0to12.yaml)
```

These are 256-world configs and expect ~90 GB of VRAM. Scale down with
`--num_envs` / `--num_agents_per_env` before running them on a smaller card.

---

## Configuration

Every experiment is one YAML file in `configs/scene_factory/`. See
**[docs/configuration.md](docs/configuration.md)** for a description of every key
in the demo config, plus how the `scene_factory.config_path` indirection to the
scene-pool file works.

Most config keys can be overridden on the command line. **The flag name is not
always the YAML leaf name** — keys under `observation:` take an `obs_` prefix,
keys under `reward:` a `reward_` prefix, and keys under `scene_factory:` a
`scene_factory_` prefix, while `env:` and `runner:` keys are used bare:

| YAML | CLI flag |
|---|---|
| `env.num_envs` | `--num_envs` |
| `env.num_agents_per_env` | `--num_agents_per_env` |
| `env.dynamics_mode` | `--dynamics_mode` |
| `observation.road_points_k` | `--obs_road_points_k` |
| `observation.weather_context_blind` | `--obs_weather_context_blind` |
| `reward.goal_bonus` | `--reward_goal_bonus` |
| `runner.max_iterations` | `--max_iterations` |

```bash
bash run_demo_train.sh --num_envs 16 --obs_road_points_k 32 --max_iterations 50
```

For the complete list: `python src/train_student_vehicle_goal_multiagent_rsl_rl.py --help`.

---

## GPU capacity benchmark

How many worlds and agents fit on your GPU, and the throughput they sustain:

```bash
PYTHONPATH=. python scripts/benchmark_casps_sweep.py --device cuda:0
```

Results depend strongly on the observation width, which the output records.
See **[docs/gpu_benchmark.md](docs/gpu_benchmark.md)** for the method, the exact
commands behind the reported sweeps, and how to read the output.

---

## Reproducing paper results

All of these need scene data (§5) and, for the 256-world configs, a large-memory
GPU.

| Experiment | Script |
|---|---|
| NHTSA-referenced braking validation | `bash run_brake_sweep_replicates.sh 10` |
| Friction conditioning ablation | `bash run_paper_table4_eval.sh` |
| Cross-dynamics transfer, 2×2 × seeds | `bash run_transfer_eval.sh` |
| GPU capacity and throughput sweep | `python scripts/benchmark_casps_sweep.py --device cuda:0` |
| Scene diversity ablation (train / eval) | `bash run_scene_diversity_ablation.sh`, `bash run_scene_diversity_eval.sh` |
| Weather-exposure training arms | `bash run_weather_ablation_rebuttal.sh`, `bash run_bicycle_dry_rebuttal.sh` |

```bash
python scripts/aggregate_table4_eval.py      # friction conditioning, across seeds
python scripts/aggregate_transfer_eval.py    # transfer matrix, across seeds
python scripts/plot_brake_sweep.py --replicates logs/rsl_rl/brake_friction_sweep/replicates_<ts>
```

The braking figure is the one to run first if you want a quick signal that the
install is sound: ten seeded sweeps take about four minutes total and need no
policy. Distances are speed-corrected to 100 km/h (SAE J299, as NHTSA did) and
compared against the measured no-ABS bands. On a correct install the dry median
lands near 57 m against a measured 48–69 m band, and wet near 69 m against
52–95 m, with 118 or so of 120 stops passing the validity filter.

Note this validates **stopping distance**, not μ. NHTSA's 0.86 and 0.66 are peak
braking coefficients from a skid trailer; the simulator's Coulomb μ is a
different quantity, and the two should not be equated.

**Which checkpoints these need.** The two policies in `checkpoints/` are the
weather-exposed / dry-trained pair, and they are what the friction-conditioning
and transfer scripts load by default — no training required. Point either at
your own run with an environment variable rather than by editing the script:

```bash
CKPT_AWARE=logs/rsl_rl/.../model_900.pt bash run_paper_table4_eval.sh
```

Two exceptions. **Braking validation** uses no policy at all — it drives the
vehicle open-loop — so it runs as soon as scene data is in place. **The transfer
matrix** needs a bicycle-backend policy, which is not shipped: train it with
`run_bicycle_dry_rebuttal.sh`, and keep it speed-matched to the PhysX vehicle
(`bicycle_max_speed_mps: 4.5`) or the comparison is between two different
problems.

### Friction model validation

```bash
PYTHONPATH=. python -m pytest src/trfc/tests/test_friction_water_film.py -q
PYTHONPATH=. python scripts/trfc_paper_validation.py --out artifacts/trfc_validation
```

The module implements the modified ALL model of Zhao et al. (2024). Two of its
parameters are not defined anywhere in that paper — the Eq. (12) coefficient `A`
and the Stribeck exponent α — so SceneFactory fixes them by calibrating against
the paper's own Fig. 6 and Tables 2–3: **`A = 4.05`** (dimensionless, established
by dimensional analysis of Eq. 12) and **`α = 0.90`**. RMSE against the published
curves is 0.0405 and 0.0206. Both are fields on `AllWetRoadParameters`, not
constants — sweep them if you have surface data of your own.

See **[docs/friction_model.md](docs/friction_model.md)** for the derivation, the
calibration procedure, and the one acceptance criterion that does not pass
because the source paper is internally inconsistent.

---

## Vehicle system identification

Pre-fitted parameters are in
`artifacts/student_vehicle_sysid/comprehensive_fwd_v1_cem_v4/best_config.json`
and load automatically. To re-run from scratch:

```bash
# 1. generate teacher maneuver programs
PYTHONPATH=. python -m src.physx_teacher_command_program_generator \
  --output-dir artifacts/physx_teacher_programs

# 2. record teacher rollouts
PYTHONPATH=. python -m src.physx_teacher_dataset_builder \
  --dataset-dir artifacts/physx_teacher_datasets/comprehensive_fwd_v1 \
  --suite sysid-comprehensive-fwd --headless

# 3. CEM fitting
PYTHONPATH=. python -m src.student_vehicle_sysid --headless \
  --teacher-dataset-manifest artifacts/physx_teacher_datasets/comprehensive_fwd_v1/manifest.json \
  --student-usd artifacts/student_vehicle_assets/vehicle_student/student_fwd_vehicle.usd \
  --output-dir artifacts/student_vehicle_sysid/my_run \
  --search-mode staged --optimizer cem
```

The vehicle is calibrated against Isaac Sim's built-in PhysX Vehicle model, not
against measured real-vehicle data.

---

## Release scope

See **[RELEASE.md](RELEASE.md)** for exactly what is and is not included.

---

## Repository structure

```
SceneFactory/
├── src/
│   ├── trfc/                                     # friction API, lane sampler, obs helpers
│   │   └── tests/                                # friction regression tests
│   ├── student_vehicle_multiagent_goal_env.py    # main RL environment
│   ├── scene_factory_multiworld_scene.py         # multi-world USD scene builder
│   ├── scene_factory_obs_contract.py             # observation space
│   ├── scene_factory_late_fusion_actor_critic.py # policy network
│   ├── scene_factory_config_wizard.py            # interactive scene-pool curation
│   ├── procedural_student_vehicle*.py            # vehicle URDF/USD generation
│   ├── student_vehicle_sysid.py                  # CEM system identification
│   ├── scene_factory_scenario_api.py             # runtime obstacles / keep-out / zone speed
│   ├── scene_factory_od_registry.py              # pluggable origin-destination modes
│   ├── braking_validation.py                     # NHTSA-referenced braking
│   ├── physics_validation.py, traction_probe.py  # physics sanity checks
│   ├── trfc/tests/                               # CPU-only regression tests
│   ├── train_student_vehicle_goal_multiagent_rsl_rl.py
│   └── eval_student_vehicle_goal_ppo.py
├── configs/scene_factory/                        # every documented config lives here
│   ├── demo_weather_physx_train.yaml             # the demo config
│   ├── demo_weather_scenes.yaml                  # its scene pool
│   └── waymo_physx_256_train_16agents_knn_*.yaml # the weather-exposure arms
├── artifacts/                                    # vehicle USD/URDF + sysid result
├── checkpoints/                                  # pre-trained policies
├── docs/
│   ├── configuration.md                          # every config key
│   ├── friction_model.md                         # weather-to-friction model
│   └── gpu_benchmark.md                          # capacity/throughput sweep
├── scripts/
│   ├── check_install.py                          # pre-flight check
│   ├── convert_waymo_tfrecord_to_json.py         # TFRecord → scene JSON
│   └── trfc_paper_validation.py
└── run_*.sh
```

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `git checkout v0.54.3` → `pathspec ... did not match` | 0.54.3 is the Python package version, not a git tag. Use commit `e10c302cd2` (§3). |
| `FileNotFoundError: scene_json does not exist: .../scene_000000.json` | No scene data. Do §5. `check_install.py` reports this before Isaac Sim starts. |
| `ModuleNotFoundError: stable_baselines3` on any eval script | `pip install -r requirements.txt`. |
| `ModuleNotFoundError: isaaclab` / `isaaclab_rl` | IsaacLab is not a sibling of this repo, or §3 was skipped. Run `check_install.py`. |
| CUDA OOM at startup | Lower `--num_envs`, `--num_agents_per_env`, and `observation.road_points_k`. |
| `pip` prints conflicts for `starlette` / `typing_extensions` after step 3 | Expected. Isaac Sim's `fastapi` pin and Isaac Lab's `starlette` pin disagree upstream; the conflict appears whichever order you install in. We did not observe it preventing installation, and have not traced it further. Install Isaac Sim **before** Isaac Lab (§2 then §3) so Isaac Lab's newer pins win. |
| `Unable to bootstrap inner kit kernel: ... GLIBCXX_3.4.30 not found` | A `libstdc++.so.6` earlier on `LD_LIBRARY_PATH` shadows your environment's — most often a *different* Anaconda install (a giveaway is `bash` itself warning about `libtinfo.so.6`). Isaac Sim binds the first one found. Fix: remove that entry, or put your env first — `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"`. If your env's own copy is also too old: `conda install -c conda-forge 'libstdcxx-ng>=12'`. `check_install.py` detects this. |
| Isaac Sim fails to start with a driver error | Check your driver against the [Isaac Sim 5.1.0 requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html). We verified on 580.173. |

---

## Citation

> Citation information will be provided upon paper acceptance.

---

## License

[MIT License](LICENSE)

The Waymo Open Motion Dataset is subject to its own [license terms](https://waymo.com/open/terms/).
NVIDIA Isaac Sim is used under NVIDIA's [non-commercial research license](https://developer.nvidia.com/isaac-sim).
