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

- **127× throughput** over a non-vectorized PhysX baseline — up to **19,250 CASPS** at 256 worlds × 16 agents on a single GPU
- **Waymo Open Motion Dataset** scenes converted to USD road environments offline; diverse topologies loaded at runtime with no code changes
- **Articulated PhysX vehicle** (10-DOF rigid body) with system-identified dynamics, fully accessible as GPU tensors
- **Weather-to-friction module** implementing the modified ALL (Average Lumped LuGre) model — maps precipitation + road surface to per-world PhysX friction coefficients
- **Bicycle-model backend** (`dynamics_mode: bicycle`) for fast prototyping and transfer studies, sharing the same observation/action/reward interface as PhysX
- **Single YAML hierarchy** controls everything: scene pool, parallelism, weather distribution, dynamics backend, and all reward weights

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
(scenario protos, v1.2; accepting the Waymo licence is required) and place one or
more `*.tfrecord` shards in `data/waymo_tfrecords/`.

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
| `checkpoints/v7_weather_aware_iter600.pt` | Weather-aware policy — conditions on friction token |
| `checkpoints/v8_no_weather_iter300.pt` | No-weather baseline — friction token masked, same architecture |

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
# PhysX, dry, 128 unique scenes
PYTHONPATH=. python src/train_student_vehicle_goal_multiagent_rsl_rl.py \
  --config configs/scene_factory/generated/scene_factory_256scene_random_0414_train_fastgoal_v8_sysid4_noweather.yaml \
  --headless

# friction-aware, 10 % wet-world exposure
PYTHONPATH=. python src/train_student_vehicle_goal_multiagent_rsl_rl.py \
  --config configs/scene_factory/generated/scene_factory_256scene_random_0414_train_fastgoal_v7_sysid4_weather.yaml \
  --headless

# bicycle backend, for transfer ablation
bash run_bicycle_train.sh --headless
```

These use the paper's 256-world configs and expect ~90 GB of VRAM. Scale down
with `--num_envs` / `--num_agents_per_env` before running them on a smaller card.

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

## Reproducing paper results

All paper experiments use the checkpoints in `checkpoints/` — no download
required. They still need scene data (§5) and, for the 256-world configs, a
large-memory GPU.

| Experiment | Script |
|---|---|
| Physics-gap cross-evaluation (Table 2) | `bash run_v8_vs_v7_physics_blind_eval.sh` |
| Friction conditioning ablation — moderate wet (Table 3) | `bash run_v8_vs_v7_moderate_wet_eval.sh` |
| Friction conditioning ablation — heavy wet | `bash run_v8_vs_v7_heavy_wet_eval.sh` |
| Bicycle → PhysX transfer | `bash run_bicycle_physx_transfer_eval.sh` |
| PhysX → Bicycle transfer | `bash run_v8_physx_to_bicycle_transfer_eval.sh` |
| Scene diversity ablation (train) | `bash run_scene_diversity_ablation.sh` |
| Scene diversity ablation (eval) | `bash run_scene_diversity_eval.sh` |

```bash
python scripts/summarize_2x2_eval.py     # summarize the 2×2 friction results
```

### Friction model validation

```bash
PYTHONPATH=. python -m pytest src/trfc/tests/test_friction_water_film.py -q
PYTHONPATH=. python scripts/trfc_paper_validation.py --out artifacts/trfc_validation
```

**The friction model was corrected after the paper was submitted.** Coefficient
`A` in Eq. (12) of the source paper was being passed a contact-patch area, which
is dimensionally inconsistent and forced μ to exactly zero above a 0.8 mm water
film. See **[docs/friction_model.md](docs/friction_model.md)** for the derivation,
the calibration, and the one acceptance criterion that still does not pass.

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
│   ├── train_student_vehicle_goal_multiagent_rsl_rl.py
│   └── eval_student_vehicle_goal_ppo.py
├── configs/scene_factory/
│   ├── demo_weather_physx_train.yaml             # the demo config
│   ├── demo_weather_scenes.yaml                  # its scene pool
│   └── generated/                                # per-experiment configs
├── artifacts/                                    # vehicle USD/URDF + sysid result
├── checkpoints/                                  # pre-trained policies
├── docs/
│   ├── configuration.md                          # every config key
│   └── friction_model.md                         # weather-to-friction model
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
| Isaac Sim fails to start with a driver error | Check your driver against the [Isaac Sim 5.1.0 requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html). We verified on 580.173. |

---

## Citation

> Citation information will be provided upon paper acceptance.

---

## License

[MIT License](LICENSE)

The Waymo Open Motion Dataset is subject to its own [license terms](https://waymo.com/open/terms/).
NVIDIA Isaac Sim is used under NVIDIA's [non-commercial research license](https://developer.nvidia.com/isaac-sim).
