# Configuration reference

Every SceneFactory experiment is described by **two** YAML files:

1. a **training config** — `configs/scene_factory/demo_weather_physx_train.yaml`
   — parallelism, observation, reward, policy, PPO, assets;
2. a **scene-pool config** — `configs/scene_factory/demo_weather_scenes.yaml` —
   which road scene and which weather each world gets.

The first points at the second through `scene_factory.config_path`. Nothing else
links them, and the path is resolved **relative to the process working
directory**, which is why everything must be run from the repository root with
`PYTHONPATH=.`.

```
demo_weather_physx_train.yaml
  └── scene_factory.config_path ──► demo_weather_scenes.yaml
                                      ├── io.scene_json_dir ──► data/processed/waymo_scenes_json/
                                      └── world.assignments[i] = {scene_json, friction}
                                                                   │
                                                    world i gets that scene and that μ
```

## Overriding from the command line

The flag name is **not** always the YAML leaf name. Keys under `observation:`
take an `obs_` prefix, `reward:` a `reward_` prefix, `scene_factory:` a
`scene_factory_` prefix; `env:` and `runner:` keys are used bare.

```bash
bash run_demo_train.sh --num_envs 16 --obs_road_points_k 32 --max_iterations 50
```

Booleans use `argparse.BooleanOptionalAction`: pass `--obs_weather_context_blind`
to enable and `--no-obs_weather_context_blind` to disable. Passing an explicit
value (`--obs_weather_context_blind true`) is a parse error.

Full list: `python src/train_student_vehicle_goal_multiagent_rsl_rl.py --help`.

---

# The training config

## `env:` — world and episode setup

| key | demo | meaning |
|---|---|---|
| `num_envs` | 32 | Parallel worlds. Dominates VRAM and throughput. Paper: 256. |
| `num_agents_per_env` | 4 | Controlled vehicles per world. Total agent slots = `num_envs × num_agents_per_env`. Paper: 16. |
| `observation_mode` | `choco_reference` | Observation preset. `choco_reference` is the full ego+weather+road+neighbour vector; see `src/scene_factory_obs_contract.py`. |
| `env_spacing` | 400.0 | Metres between world origins on the stage grid. Must exceed the scene extent or worlds overlap visually. |
| `spawn_height_m` | 1.2 | Height agents are dropped from at reset. |
| `spawn_yaw_noise_rad` | 0.0 | Uniform yaw jitter at spawn. |
| `ground_mode` | `cuboid` | Each world gets its own 1000×1000×1 m kinematic ground cuboid, cloned from env 0. `plane` (a single shared infinite ground) is no longer supported and falls back to `cuboid` with a warning. |
| `use_scene_factory_roads` | true | Build road geometry from the scene pool. False gives an empty plane. |
| `reset_mode` | `teleport_only` | Reset strategy. |
| `start_radius_m` | 0.0 | Radius of the ring agents start on around the scene origin. |
| `agent_spawn_circle_radius_m` | 0.0 | Radius of the circle agents are distributed around within a world. |
| `agent_spawn_jitter_m` | 0.0 | Random per-agent spawn offset. |
| `episode_length_s` | 45.0 | Episode horizon in seconds. |
| `goal_radius_min_m` / `goal_radius_max_m` | 5.0 / 30.0 | Goals are sampled at this distance range from the agent's spawn. |
| `goal_reached_threshold_m` | 3.0 | Success radius. Directly sets the success-rate metric. |
| `fall_height_threshold_m` | 0.0 | Below this z, the agent is treated as fallen and reset. |
| `bad_tilt_gravity_threshold` | −0.15 | Reset if the projected gravity indicates a rollover. |
| `max_distance_from_origin_m` | 100.0 | Reset if the agent leaves this radius. Bounds runaway agents. |
| `agent_neighbor_obs_scale_m` | 100.0 | Normalizing distance for neighbour relative positions. |
| `agent_collision_warmup_steps` | 24 | Steps after reset during which agent–agent collisions are not penalized (avoids punishing overlapping spawns). |
| `replicate_physics` | false | Isaac Lab physics replication. Must be false when worlds differ. |
| `clone_in_fabric` | `"false"` | Clone through Fabric rather than USD. |
| `dynamics_mode` | `physx` (default) | `physx` = articulated 10-DOF sysid-calibrated vehicle; `bicycle` = kinematic bicycle model with the same observation/action interface. |

> **How per-world friction avoids cross-world interaction.** The μ from
> `estimate_friction()` is **not** written to the ground. It is written to the
> **wheel-shape materials of that world's own vehicles**, per environment index,
> via `root_physx_view.set_material_properties()` (see
> `src/student_vehicle_multiagent_goal_env.py`). The ground keeps a single fixed
> material and PhysX combines ground × wheel per contact. Worlds therefore stay
> independent even on the shared infinite `plane`, which is what the demo uses.

## `scene_factory:` — which scenes

| key | demo | meaning |
|---|---|---|
| `config_path` | `configs/scene_factory/demo_weather_scenes.yaml` | The scene-pool file. **Relative to the working directory.** |
| `world_index` | 0 | Which pool entry world 0 uses when selection is not random. |
| `world_selection_mode` | `random_envs` | `random_envs` draws each world's scene from the pool; otherwise worlds follow `world_index` order. |
| `random_world_seed` | 42 | Seed for that draw. Change it to reshuffle the scene→world mapping without editing the pool. |

## `observation:` — what the policy sees

The observation is `[ego | weather | road points | neighbours]`, defined in
`src/scene_factory_obs_contract.py`.

| key | demo | meaning |
|---|---|---|
| `weather_context_enable` | true | Append the 4-D weather token `[h_w/1.0, 1_AC, 1_SMA, 1_OGFC]`. |
| `weather_context_blind` | (commented) | If true, always feed the **dry-AC** token regardless of the world's real friction. This is the ablation used for the weather-conditioning study — the environment is still wet, the policy just cannot see it. |
| `road_points_enable` | true | Include nearest road centre points. |
| `road_points_k` | 64 | Number of road points. **Second-largest VRAM term after `num_envs`.** Paper: 350. |
| `road_points_radius_m` | 10.0 | Search radius for road points. |
| `road_points_type_norm` | 20.0 | Divisor normalizing the Waymo polyline type code into the observation. |
| `road_points_mode` | `road_running` | Selection strategy for which points enter the observation. |
| `road_points_include_dirs` | true | Append each point's tangent direction (3 → 5 floats per point). |
| `neighbor_enable` | true | Include other vehicles. |
| `neighbor_k` | 8 | Neighbour slots per agent. Paper: 24. |
| `neighbor_include_ttc` | true | Append time-to-collision per neighbour. |
| `neighbor_include_index` | false | Append the neighbour's agent index. |
| `neighbor_ttc_max_s` | 10.0 | TTC clamp. |
| `timing_print_enable` / `timing_print_every_n` | false / 32 | Per-phase observation timing printouts. |

> **Note.** Road observations are brute-force k-NN lookups against a *static*
> privileged map loaded at scene build time — not perception. See "Known
> limitations" in `RELEASE.md`.

## `timing:` — profiling

| key | demo | meaning |
|---|---|---|
| `step_log_enable` | true | Record per-step phase timings. |
| `step_print_enable` | false | Print them. |
| `step_print_every_n` | 128 | Print interval. |
| `step_cuda_sync_enable` | false | Insert `cuda.synchronize()` around phases. Needed for *accurate* per-phase numbers, but it serializes the pipeline and lowers throughput — leave off for performance runs. |

## `reward:` — reward shaping

`mode: choco_baseline` selects the reward used in the paper. Terms are summed per
step; positive values are bonuses, negative are penalties.

| key | demo | meaning |
|---|---|---|
| `goal_bonus` | 60.0 | One-off reward for reaching the goal. |
| `lane_center_enable` | false | Dense reward for lane-centre proximity (superseded by the `choco_geom_lane` terms). |
| `lane_forbidden_enable` | true | Penalize entering forbidden polyline types (road edge / median, Waymo type codes 15 and 16). |
| `lane_forbidden_penalty` | −20.0 | Magnitude of that penalty. |
| `collision_penalty` | −6.0 | Agent–agent contact. |
| `crash_penalty` | −10.0 | Terminating crash. |
| `choco_offroad_penalty` | −0.5 | Per-step off-road penalty. |
| `choco_idle_penalty_enable` | true | Penalize standing still, to stop the degenerate "park and survive" policy. |
| `choco_idle_penalty_per_step` | 0.06 | Its magnitude. |
| `choco_idle_speed_threshold_mps` | 2.5 | Below this speed the agent counts as idle. |
| `choco_geom_lane_enable` | true | Geometric lane-following reward. |
| `choco_geom_lane_per_step` | 0.08 | Its weight. |
| `choco_geom_lane_tolerance_m` | 1.75 | Lateral offset treated as "in lane". |
| `choco_geom_lane_heading_weight` | 0.8 | How much heading alignment counts vs lateral offset. |
| `choco_geom_lane_min_alignment` | 0.35 | Alignment below which no lane reward is given. |
| `choco_geom_route_progress_weight` | 4.0 | Reward per metre of progress toward the goal. **The main learning signal.** |
| `choco_geom_offroad_enable` | true | Geometric off-road detection. |
| `choco_geom_offroad_lateral_threshold_m` | 3.25 | Lateral distance from lane centre counted as off-road. |
| `choco_geom_offroad_distance_threshold_m` | 6.0 | Distance to nearest road point counted as off-road. |
| `choco_ttc_penalty_enable` | true | Penalize low time-to-collision with other agents. |
| `choco_ttc_penalty_alpha` | 0.10 | Its scale. |
| `choco_ttc_penalty_max` | 0.35 | Per-step cap. |
| `choco_ttc_penalty_min_ttc` | 0.5 | TTC below which the penalty saturates. |
| `choco_road_edge_ttc_penalty_*` | see config | Same construction, against road edges instead of vehicles. `alpha` 0.07, `max` 0.40, `min_ttc` 0.5, `hard_min_ttc` 0.5, `radius_m` 40.0. |

## `policy:` — network architecture

`src/scene_factory_late_fusion_actor_critic.py`. Separate encoders for ego, road
and neighbour segments; each set encoding is pooled and concatenated into a shared
trunk.

| key | demo | meaning |
|---|---|---|
| `type` | `late_fusion` | Registered custom RSL-RL actor-critic. |
| `ego_layers` | [64, 64] | Ego branch MLP. |
| `road_layers` | [96, 96] | Road-point branch, applied per point. |
| `vehicle_layers` | [96, 96] | Neighbour branch, applied per neighbour. |
| `shared_layers` | [128, 64] | Trunk after fusion. |
| `last_layer_dim_pi` / `last_layer_dim_vf` | 64 / 64 | Actor / critic head widths. |
| `activation` | `relu` | |
| `dropout` | 0.0 | |
| `pool` | `max` | Permutation-invariant pooling over the road and neighbour sets. Max-pool makes the policy independent of point/neighbour ordering. |

## `assets:` — vehicle

| key | demo | meaning |
|---|---|---|
| `student_usd` | `artifacts/.../student_fwd_vehicle.usd` | Articulated vehicle USD. Required — `check_install.py` verifies it. |
| `tunable_config_json` | `artifacts/student_vehicle_sysid/comprehensive_fwd_v1_cem_v4/best_config.json` | CEM-fitted dynamics parameters, applied at env startup. |

## `runner:` — PPO and logging

| key | demo | meaning |
|---|---|---|
| `log_dir` | `logs/rsl_rl` | Root for run directories. |
| `seed` | 42 | Seeds torch/numpy and the env. |
| `experiment_name` / `run_name` | `scene_factory_demo` / `demo_weather_physx` | Output goes to `<log_dir>/<experiment_name>/<run_name>/`. |
| `shared_policy_mode` | `agent_slots` | All agent slots share one policy; each slot is an independent sample. |
| `max_iterations` | 500 | PPO iterations. |
| `num_steps_per_env` | 64 | Rollout length. Batch = `num_envs × num_agents_per_env × num_steps_per_env`. |
| `save_interval` | 50 | Checkpoint every N iterations. |
| `learning_rate` | 3.0e-4 | Adaptive — `desired_kl` moves it. |
| `num_learning_epochs` | 5 | Passes per batch. |
| `num_mini_batches` | 4 | Minibatches per epoch. |
| `entropy_coef` | 3.0e-5 | Exploration bonus. |
| `clip_param` | 0.15 | PPO clip. |
| `desired_kl` | 0.01 | Target KL for the adaptive LR schedule. |
| `gamma` | 0.99 | Discount. |
| `gae_lambda` | 0.98 | GAE. |
| `value_loss_coef` | 1.0 | Value-loss weight. |

## `app:` — devices

| key | demo | meaning |
|---|---|---|
| `device` | `cuda:0` | Simulation device. |
| `rl_device` | `cuda:0` | Learning device. Keep equal to `device` to avoid host round-trips. |

---

# The scene-pool config

`configs/scene_factory/demo_weather_scenes.yaml`.

## `io:`

| key | meaning |
|---|---|
| `scene_json_dir` | Directory of converted Waymo scene JSONs. Default `data/processed/waymo_scenes_json`. |
| `scene_jsons` | Optional explicit filename list, used instead of `world.assignments`. Must have at least `world_count` entries. |

## `world:`

| key | demo | meaning |
|---|---|---|
| `root_container` | `/World/SceneFactoryWorlds` | USD prim path all worlds live under. |
| `world_count` | 32 | Must match `env.num_envs`. |
| `grid_cols` / `rows` | 8 / 4 | Stage grid layout. |
| `world_size_m` | [200, 200] | Nominal per-world footprint. |
| `padding_m` | 200.0 | Gap between worlds on the grid. |
| `bounds_size_m` | 200.0 | Scene crop. Polylines outside are **dropped at build time** — a scene's JSON can contain roads the environment never sees. |
| `base_z_m` | 0.0 | Ground height. |
| `origin_mode` / `origin_center_mode` | `center` / `mean` | How a scene is recentred on its world origin. |
| `assignment_fill_mode` | `random_fill` | `strict` requires exactly `world_count` assignments; `random_fill` shuffles and repeats the given list to fill `world_count`. |
| `assignment_fill_seed` | 2024 | Seed for that fill. |
| `assignments` | list | One entry per world (or fewer, with `random_fill`). |

### `world.assignments[i]`

```yaml
- scene_json: scene_000000.json
  friction: {road_type: AC, precip_type: clear, precip_intensity_mmph: 0.0, water_film_mm: 0.0}
```

| field | meaning |
|---|---|
| `scene_json` | Filename inside `io.scene_json_dir`. Missing files raise `FileNotFoundError: scene_json does not exist: …` **after** Isaac Sim boots — run `check_install.py` first. |
| `friction.road_type` | `AC` (asphalt concrete), `SMA` (stone mastic asphalt), or `OGFC` (open-graded friction course). Also one-hot encoded into the weather observation. |
| `friction.precip_type` | `clear` or `rain`. |
| `friction.precip_intensity_mmph` | Rainfall rate. |
| `friction.water_film_mm` | Water-film depth. **This is the variable that actually drives μ.** |

The friction block is passed to `estimate_friction()`
(`src/trfc/friction_api.py`), and the resulting scalar μ is applied as that
world's PhysX ground friction. The demo's four groups are dry AC (0.0 mm), light
SMA (0.3 mm), moderate SMA (0.5 mm) and heavy SMA (2.0 mm).

See [friction_model.md](friction_model.md) — including the post-submission
correction to Eq. (12), which changes μ above ~0.8 mm.

## `ground:` (optional)

| key | meaning |
|---|---|
| `friction_pipeline.enable` | If true, **every** assignment must carry a `friction` block or config loading fails. |
| `friction_pipeline.defaults` | Default `FrictionInput` fields merged under each per-world block. |

---

## Making your own experiment

1. Copy `demo_weather_physx_train.yaml` and `demo_weather_scenes.yaml`.
2. Point the new training config's `scene_factory.config_path` at your new pool.
3. Keep `world.world_count` equal to `env.num_envs`.
4. List the scenes you actually have — check with
   `PYTHONPATH=. python scripts/check_install.py --config <your train config>`.
5. Set `runner.experiment_name` / `run_name` so outputs do not collide.

Generated per-experiment configs live in `configs/scene_factory/generated/`.
`src/scene_factory_config_wizard.py` builds scene pools interactively inside Isaac
Sim.
