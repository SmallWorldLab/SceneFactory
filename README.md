# SceneFactory

**SceneFactory** is a GPU-vectorized platform for procedural scene construction, physics-based multi-agent simulation, and reinforcement learning in autonomous driving environments.

Built on [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) and [Isaac Lab](https://github.com/isaac-sim/IsaacLab), SceneFactory represents worlds and agents as batched tensors — vehicle control, observations, rewards, resets, and policy inference are all GPU tensor operations.

> Paper: *SceneFactory: GPU-Accelerated Multi-Agent Driving Simulation with Physics-Based Vehicle Dynamics* (NeurIPS 2026)

---

## Key Features

- **127× throughput** over a non-vectorized PhysX baseline — up to **19,250 CASPS** at 256 worlds × 16 agents on a single GPU
- **Waymo Open Motion Dataset** scenes converted to USD road environments offline; diverse topologies loaded at runtime with no code changes
- **Articulated PhysX vehicle** (10-DOF rigid body) with system-identified dynamics, fully accessible as GPU tensors
- **Weather-to-friction module** implementing the modified ALL (Average Lumped LuGre) model — maps precipitation + road surface to per-world PhysX friction coefficients
- **Bicycle-model backend** (`backend: bicycle`) for fast prototyping and transfer studies, sharing the same observation/action/reward interface as PhysX
- **Single YAML hierarchy** controls everything: scene pool, parallelism, weather distribution, dynamics backend, and all reward weights

---

## Requirements

| Component | Version |
|-----------|---------|
| NVIDIA Isaac Sim Full | 5.1.0 |
| Isaac Lab | 0.54.3 |
| RSL-RL | 3.1.2 |
| PyTorch | 2.7.0+cu128 |
| CUDA | 12.8 |
| GPU (tested) | NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM) |

Isaac Sim can be installed as a pip package inside a conda environment (Python 3.11 recommended).

---

## Installation

### 1. Create a conda environment

```bash
conda create -n scenefactory python=3.11 -y
conda activate scenefactory
```

### 2. Install Isaac Sim via pip

```bash
pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com
pip install isaacsim-rl==5.1.0 --extra-index-url https://pypi.nvidia.com
```

> **Note:** The first install pulls ~8 GB of Isaac Sim extensions. Subsequent runs use the cached extensions.

### 3. Install Isaac Lab from source

Isaac Lab 0.54.3 is not on PyPI — install it directly from the GitHub repo:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v0.54.3
pip install -e source/isaaclab
pip install -e source/isaaclab_assets
pip install -e source/isaaclab_tasks
cd ..
```

### 4. Install RSL-RL and other dependencies

```bash
pip install rsl-rl==3.1.2
```

### 5. Clone this repo

```bash
git clone https://github.com/BrainCrackLab/SceneFactory.git
cd SceneFactory
```

### 6. Prepare Waymo scene data

Download the [Waymo Open Motion Dataset](https://waymo.com/open/data/motion/) TFRecords, then run the offline extraction pipeline to produce per-scenario JSON files:

```bash
python -m src.trfc.world_pipeline \
  --tfrecord-dir /path/to/waymo_tfrecords \
  --output-dir data/processed/waymo_scenes_json
```

Point all configs at the output directory via `scene_json_dir: data/processed/waymo_scenes_json` (this is already the default in the provided configs).

---

## Quickstart

### Visualize scenes (headless-optional)

```bash
bash run_visualize_scene.sh --world_count 4
```

### Train (PhysX, dry, 128 unique scenes)

```bash
PYTHONPATH=. python src/train_student_vehicle_goal_multiagent_rsl_rl.py \
  --config configs/scene_factory/generated/scene_factory_256scene_random_0414_train_fastgoal_v8_sysid4_noweather.yaml \
  --headless
```

### Train (friction-aware, 10 % wet-world exposure)

```bash
PYTHONPATH=. python src/train_student_vehicle_goal_multiagent_rsl_rl.py \
  --config configs/scene_factory/generated/scene_factory_256scene_random_0414_train_fastgoal_v7_sysid4_weather.yaml \
  --headless
```

### Train (bicycle backend, for transfer ablation)

```bash
bash run_bicycle_train.sh --headless
```

---

## Reproducing Paper Results

All paper experiments use pre-trained checkpoints. Download them from [TODO: add release link] and place them under `runs/`.

| Experiment | Script |
|---|---|
| Physics-gap cross-evaluation (Table 2) | `bash run_v8_vs_v7_physics_blind_eval.sh` |
| Friction conditioning ablation — moderate wet (Table 3) | `bash run_v8_vs_v7_moderate_wet_eval.sh` |
| Friction conditioning ablation — heavy wet | `bash run_v8_vs_v7_heavy_wet_eval.sh` |
| Bicycle → PhysX transfer | `bash run_bicycle_physx_transfer_eval.sh` |
| PhysX → Bicycle transfer | `bash run_v8_physx_to_bicycle_transfer_eval.sh` |
| Scene diversity ablation (train) | `bash run_scene_diversity_ablation.sh` |
| Scene diversity ablation (eval) | `bash run_scene_diversity_eval.sh` |

Summarize 2×2 friction results:
```bash
python scripts/summarize_2x2_eval.py
```

---

## Vehicle System Identification

The pre-fitted sysid parameters are in `artifacts/student_vehicle_sysid/comprehensive_fwd_v1_cem_v4/best_config.json` and are loaded automatically.

To re-run sysid from scratch:

**Step 1 — Generate teacher maneuver programs:**
```bash
PYTHONPATH=. python -m src.physx_teacher_command_program_generator \
  --output-dir artifacts/physx_teacher_programs
```

**Step 2 — Record teacher rollouts:**
```bash
PYTHONPATH=. python -m src.physx_teacher_dataset_builder \
  --dataset-dir artifacts/physx_teacher_datasets/comprehensive_fwd_v1 \
  --suite sysid-comprehensive-fwd \
  --headless
```

**Step 3 — Run CEM fitting:**
```bash
PYTHONPATH=. python -m src.student_vehicle_sysid \
  --headless \
  --teacher-dataset-manifest artifacts/physx_teacher_datasets/comprehensive_fwd_v1/manifest.json \
  --student-usd artifacts/student_vehicle_assets/vehicle_student/student_fwd_vehicle.usd \
  --output-dir artifacts/student_vehicle_sysid/my_run \
  --search-mode staged \
  --optimizer cem
```

---

## Repository Structure

```
SceneFactory/
├── src/
│   ├── trfc/                          # Friction API, lane sampler, obs helpers
│   ├── student_vehicle_multiagent_goal_env.py   # Main RL environment
│   ├── scene_factory_multiworld_scene.py        # Multi-world USD scene builder
│   ├── scene_factory_obs_contract.py            # Observation space
│   ├── scene_factory_late_fusion_actor_critic.py # Policy network
│   ├── scene_factory_config_wizard.py           # Config loader
│   ├── procedural_student_vehicle*.py           # Vehicle URDF/USD generation
│   ├── student_vehicle_sysid.py                 # CEM system identification
│   ├── train_student_vehicle_goal_multiagent_rsl_rl.py   # Training entry point
│   └── eval_student_vehicle_goal_ppo.py                  # Eval entry point
├── configs/scene_factory/
│   ├── generated/                     # Per-experiment YAML configs
│   └── *.yaml                         # Canonical base configs
├── artifacts/
│   ├── student_vehicle_assets/vehicle_student/  # URDF + USD + spec
│   └── student_vehicle_sysid/                   # Final sysid result
├── scripts/
│   ├── summarize_2x2_eval.py
│   └── generate_weather_eval_configs.py
└── run_*.sh                           # Paper experiment launchers
```

---

## Citation

```bibtex
@inproceedings{zhu2026scenefactory,
  title     = {SceneFactory: GPU-Accelerated Multi-Agent Driving Simulation with Physics-Based Vehicle Dynamics},
  author    = {Zhu, Yicheng and Bian, Zilin},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

---

## License

[MIT License](LICENSE)

The Waymo Open Motion Dataset is subject to its own [license terms](https://waymo.com/open/terms/).
NVIDIA Isaac Sim is used under NVIDIA's [non-commercial research license](https://developer.nvidia.com/isaac-sim).
