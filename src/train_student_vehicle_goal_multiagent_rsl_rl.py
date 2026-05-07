from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import types
from datetime import datetime
from time import perf_counter
from typing import Any

import yaml

from src.isaaclab_bootstrap import ensure_isaaclab_source_paths

ensure_isaaclab_source_paths()

os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp_cache")

from isaaclab.app import AppLauncher


DEFAULT_CONFIG_PATH = "configs/scene_factory/goal_reaching_train.yaml"


def _load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Training config root must be a mapping, got {type(payload).__name__}")
    return payload


def _cfg_value(cfg: dict[str, Any], section: str, key: str, default: Any) -> Any:
    section_payload = cfg.get(section, {}) or {}
    if not isinstance(section_payload, dict):
        return default
    return section_payload.get(key, default)


def _cfg_int_tuple(cfg: dict[str, Any], section: str, key: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = _cfg_value(cfg, section, key, default)
    if value is None:
        return tuple(int(item) for item in default)
    if isinstance(value, str):
        tokens = [token.strip() for token in value.replace(";", ",").split(",") if token.strip()]
        return tuple(int(token) for token in tokens) if tokens else tuple(int(item) for item in default)
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    return (int(value),)


def _validate_training_config_shape(cfg: dict[str, Any], config_path: str | Path) -> None:
    required_training_sections = ("runner", "scene_factory", "assets")
    if all(isinstance(cfg.get(section), dict) for section in required_training_sections):
        return

    scene_config_markers = ("world", "road", "vehicles")
    looks_like_scene_config = all(isinstance(cfg.get(section), dict) for section in scene_config_markers)
    if looks_like_scene_config:
        raise SystemExit(
            "The provided --config appears to be a SceneFactory world/scene config, not a training config: "
            f"{Path(config_path).expanduser().resolve()}\n"
            "Use a training preset such as\n"
            "  configs/scene_factory/goal_reaching_roads_choco_obs_random4_curated32_agent_slots_goal3_late_fusion_obs_aligned_ttc_index.yaml\n"
            "which references the scene config via scene_factory.config_path."
        )


pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
pre_args, _ = pre_parser.parse_known_args()
config_path = str(Path(pre_args.config).expanduser().resolve())
file_cfg = _load_yaml_config(config_path)
_validate_training_config_shape(file_cfg, config_path)

parser = argparse.ArgumentParser(
    parents=[pre_parser],
    description="Train a shared-policy PPO controller for the multi-agent student-vehicle goal task using Isaac Lab RSL-RL."
)
parser.add_argument("--num_envs", type=int, default=int(_cfg_value(file_cfg, "env", "num_envs", 16)), help="Number of parallel world instances.")
parser.add_argument("--num_agents_per_env", type=int, default=int(_cfg_value(file_cfg, "env", "num_agents_per_env", 2)), help="Number of vehicles inside each world.")
parser.add_argument("--seed", type=int, default=int(_cfg_value(file_cfg, "runner", "seed", 42)), help="Random seed.")
parser.add_argument("--max_iterations", type=int, default=int(_cfg_value(file_cfg, "runner", "max_iterations", 10)), help="Number of RSL-RL PPO learning iterations.")
parser.add_argument("--log_dir", type=str, default=str(_cfg_value(file_cfg, "runner", "log_dir", "logs/rsl_rl")), help="Training log root.")
parser.add_argument("--student_usd", type=str, default=str(_cfg_value(file_cfg, "assets", "student_usd", "")), help="Path to the student vehicle USD.")
parser.add_argument(
    "--observation_mode",
    choices=("full", "goal_reaching", "choco_reference"),
    default=str(_cfg_value(file_cfg, "env", "observation_mode", "goal_reaching")),
    help="Policy observation preset.",
)
parser.add_argument(
    "--obs_weather_context_enable",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "observation", "weather_context_enable", True)),
    help="Include SceneFactory weather/friction context in the choco_reference observation mode.",
)
parser.add_argument(
    "--obs_weather_context_blind",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "observation", "weather_context_blind", False)),
    help="Feed all-zeros weather context regardless of actual surface friction (for OOD eval of dry-trained models on varied-friction scenes).",
)
parser.add_argument(
    "--obs_road_points_enable",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "observation", "road_points_enable", True)),
)
parser.add_argument(
    "--obs_road_points_k",
    type=int,
    default=int(_cfg_value(file_cfg, "observation", "road_points_k", 200)),
)
parser.add_argument(
    "--obs_road_points_radius_m",
    type=float,
    default=float(_cfg_value(file_cfg, "observation", "road_points_radius_m", 35.0)),
)
parser.add_argument(
    "--obs_road_points_type_norm",
    type=float,
    default=float(_cfg_value(file_cfg, "observation", "road_points_type_norm", 20.0)),
)
parser.add_argument(
    "--obs_road_points_mode",
    choices=("knn", "road_running", "road-running"),
    default=str(_cfg_value(file_cfg, "observation", "road_points_mode", "road-running")),
)
parser.add_argument(
    "--obs_road_points_include_dirs",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "observation", "road_points_include_dirs", False)),
)
parser.add_argument(
    "--obs_neighbor_enable",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "observation", "neighbor_enable", True)),
)
parser.add_argument(
    "--obs_neighbor_k",
    type=int,
    default=int(_cfg_value(file_cfg, "observation", "neighbor_k", 63)),
)
parser.add_argument(
    "--obs_neighbor_include_ttc",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "observation", "neighbor_include_ttc", False)),
)
parser.add_argument(
    "--obs_neighbor_include_index",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "observation", "neighbor_include_index", False)),
)
parser.add_argument(
    "--obs_neighbor_ttc_max_s",
    type=float,
    default=float(_cfg_value(file_cfg, "observation", "neighbor_ttc_max_s", 10.0)),
)
parser.add_argument(
    "--obs_timing_print_enable",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "observation", "timing_print_enable", False)),
)
parser.add_argument(
    "--obs_timing_print_every_n",
    type=int,
    default=int(_cfg_value(file_cfg, "observation", "timing_print_every_n", 32)),
)
parser.add_argument(
    "--step_timing_log_enable",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "timing", "step_log_enable", False)),
    help="Log per-step SceneFactory timing metrics into extras/TensorBoard.",
)
parser.add_argument(
    "--step_timing_print_enable",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "timing", "step_print_enable", False)),
    help="Print per-step SceneFactory timing breakdowns every N steps.",
)
parser.add_argument(
    "--step_timing_print_every_n",
    type=int,
    default=int(_cfg_value(file_cfg, "timing", "step_print_every_n", 128)),
)
parser.add_argument(
    "--step_timing_cuda_sync_enable",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "timing", "step_cuda_sync_enable", False)),
    help="Synchronize CUDA around step/reset timing regions for more accurate profiling at the cost of runtime overhead.",
)
parser.add_argument(
    "--reward_lane_center_enable",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "reward", "lane_center_enable", True)),
)
parser.add_argument(
    "--reward_lane_center_per_step",
    type=float,
    default=float(_cfg_value(file_cfg, "reward", "lane_center_per_step", 0.05)),
)
parser.add_argument(
    "--reward_lane_forbidden_enable",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "reward", "lane_forbidden_enable", True)),
)
parser.add_argument(
    "--reward_lane_forbidden_penalty",
    type=float,
    default=float(_cfg_value(file_cfg, "reward", "lane_forbidden_penalty", -30.0)),
)
parser.add_argument(
    "--reward_collision_penalty",
    type=float,
    default=float(_cfg_value(file_cfg, "reward", "collision_penalty", -15.0)),
)
parser.add_argument(
    "--reward_crash_penalty",
    type=float,
    default=float(_cfg_value(file_cfg, "reward", "crash_penalty", -10.0)),
)
parser.add_argument(
    "--reward_mode",
    choices=("scene_factory_default", "choco_baseline"),
    default=str(_cfg_value(file_cfg, "reward", "mode", "scene_factory_default")),
)
parser.add_argument(
    "--reward_goal_bonus",
    type=float,
    default=float(_cfg_value(file_cfg, "reward", "goal_bonus", 20.0)),
)
parser.add_argument("--reward_choco_offroad_penalty", type=float, default=float(_cfg_value(file_cfg, "reward", "choco_offroad_penalty", -0.5)))
parser.add_argument("--reward_choco_idle_penalty_enable", action=argparse.BooleanOptionalAction, default=bool(_cfg_value(file_cfg, "reward", "choco_idle_penalty_enable", True)))
parser.add_argument("--reward_choco_idle_penalty_per_step", type=float, default=float(_cfg_value(file_cfg, "reward", "choco_idle_penalty_per_step", 0.03)))
parser.add_argument("--reward_choco_idle_speed_threshold_mps", type=float, default=float(_cfg_value(file_cfg, "reward", "choco_idle_speed_threshold_mps", 0.5)))
parser.add_argument("--reward_choco_speed_bonus_enable", action=argparse.BooleanOptionalAction, default=bool(_cfg_value(file_cfg, "reward", "choco_speed_bonus_enable", False)))
parser.add_argument("--reward_choco_speed_bonus_per_step", type=float, default=float(_cfg_value(file_cfg, "reward", "choco_speed_bonus_per_step", 0.02)))
parser.add_argument("--reward_choco_speed_bonus_max_mps", type=float, default=float(_cfg_value(file_cfg, "reward", "choco_speed_bonus_max_mps", 10.0)))
parser.add_argument("--reward_choco_geom_lane_enable", action=argparse.BooleanOptionalAction, default=bool(_cfg_value(file_cfg, "reward", "choco_geom_lane_enable", True)))
parser.add_argument("--reward_choco_geom_lane_per_step", type=float, default=float(_cfg_value(file_cfg, "reward", "choco_geom_lane_per_step", 0.12)))
parser.add_argument("--reward_choco_geom_lane_tolerance_m", type=float, default=float(_cfg_value(file_cfg, "reward", "choco_geom_lane_tolerance_m", 1.75)))
parser.add_argument("--reward_choco_geom_lane_heading_weight", type=float, default=float(_cfg_value(file_cfg, "reward", "choco_geom_lane_heading_weight", 0.8)))
parser.add_argument("--reward_choco_geom_lane_min_alignment", type=float, default=float(_cfg_value(file_cfg, "reward", "choco_geom_lane_min_alignment", 0.35)))
parser.add_argument("--reward_choco_geom_route_progress_weight", type=float, default=float(_cfg_value(file_cfg, "reward", "choco_geom_route_progress_weight", 0.0)))
parser.add_argument("--reward_choco_geom_offroad_enable", action=argparse.BooleanOptionalAction, default=bool(_cfg_value(file_cfg, "reward", "choco_geom_offroad_enable", True)))
parser.add_argument("--reward_choco_geom_offroad_lateral_threshold_m", type=float, default=float(_cfg_value(file_cfg, "reward", "choco_geom_offroad_lateral_threshold_m", 3.25)))
parser.add_argument("--reward_choco_geom_offroad_distance_threshold_m", type=float, default=float(_cfg_value(file_cfg, "reward", "choco_geom_offroad_distance_threshold_m", 6.0)))
parser.add_argument("--reward_choco_ttc_penalty_enable", action=argparse.BooleanOptionalAction, default=bool(_cfg_value(file_cfg, "reward", "choco_ttc_penalty_enable", True)))
parser.add_argument("--reward_choco_ttc_penalty_alpha", type=float, default=float(_cfg_value(file_cfg, "reward", "choco_ttc_penalty_alpha", 0.15)))
parser.add_argument("--reward_choco_ttc_penalty_max", type=float, default=float(_cfg_value(file_cfg, "reward", "choco_ttc_penalty_max", 0.5)))
parser.add_argument("--reward_choco_ttc_penalty_min_ttc", type=float, default=float(_cfg_value(file_cfg, "reward", "choco_ttc_penalty_min_ttc", 0.5)))
parser.add_argument("--reward_choco_road_edge_ttc_penalty_enable", action=argparse.BooleanOptionalAction, default=bool(_cfg_value(file_cfg, "reward", "choco_road_edge_ttc_penalty_enable", True)))
parser.add_argument("--reward_choco_road_edge_ttc_penalty_alpha", type=float, default=float(_cfg_value(file_cfg, "reward", "choco_road_edge_ttc_penalty_alpha", 0.10)))
parser.add_argument("--reward_choco_road_edge_ttc_penalty_max", type=float, default=float(_cfg_value(file_cfg, "reward", "choco_road_edge_ttc_penalty_max", 0.60)))
parser.add_argument("--reward_choco_road_edge_ttc_penalty_min_ttc", type=float, default=float(_cfg_value(file_cfg, "reward", "choco_road_edge_ttc_penalty_min_ttc", 0.5)))
parser.add_argument("--reward_choco_road_edge_ttc_hard_min_ttc", type=float, default=float(_cfg_value(file_cfg, "reward", "choco_road_edge_ttc_hard_min_ttc", 0.5)))
parser.add_argument("--reward_choco_road_edge_ttc_radius_m", type=float, default=float(_cfg_value(file_cfg, "reward", "choco_road_edge_ttc_radius_m", 40.0)))
parser.add_argument(
    "--tunable_config_json",
    type=str,
    default=str(_cfg_value(file_cfg, "assets", "tunable_config_json", "")),
    help="Path to the tuned student config JSON. Empty uses the environment default.",
)
parser.add_argument("--spawn_height_m", type=float, default=float(_cfg_value(file_cfg, "env", "spawn_height_m", 1.6)), help="Vehicle spawn height above each env origin.")
parser.add_argument(
    "--spawn_yaw_noise_rad",
    type=float,
    default=float(_cfg_value(file_cfg, "env", "spawn_yaw_noise_rad", 0.5)),
    help="Uniform random yaw perturbation added at spawn.",
)
parser.add_argument(
    "--decimation",
    type=int,
    default=int(_cfg_value(file_cfg, "env", "decimation", 4)),
    help="Number of physics substeps per environment step.",
)
parser.add_argument(
    "--ground_mode",
    choices=("plane", "cuboid"),
    default=str(_cfg_value(file_cfg, "env", "ground_mode", "plane")),
    help="Ground implementation for the training scene.",
)
parser.add_argument(
    "--apply_runtime_external_wrench",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "env", "apply_runtime_external_wrench", True)),
    help="Apply the runtime lateral/yaw damping wrench each physics substep.",
)
parser.add_argument(
    "--use_scene_factory_roads",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "env", "use_scene_factory_roads", False)),
    help="Build SceneFactory road geometry inside env_0 and clone it across vectorized worlds.",
)
parser.add_argument(
    "--scene_factory_config",
    type=str,
    default=str(_cfg_value(file_cfg, "scene_factory", "config_path", "configs/scene_factory/multiworld_scene.yaml")),
    help="SceneFactory YAML config used to source road geometry and scenario start/goal pairs.",
)
parser.add_argument(
    "--scene_factory_world_index",
    type=int,
    default=int(_cfg_value(file_cfg, "scene_factory", "world_index", 0)),
    help="Which resolved SceneFactory world to use as the cloned source world.",
)
parser.add_argument(
    "--scene_factory_world_selection_mode",
    choices=("fixed", "random_envs", "random-envs"),
    default=str(_cfg_value(file_cfg, "scene_factory", "world_selection_mode", "fixed")),
    help="How SceneFactory worlds are assigned across vectorized environments.",
)
parser.add_argument(
    "--scene_factory_random_world_seed",
    type=int,
    default=int(_cfg_value(file_cfg, "scene_factory", "random_world_seed", 42)),
    help="Seed used when SceneFactory world_selection_mode=random_envs.",
)
parser.add_argument(
    "--reset_mode",
    choices=("isaac_reset", "teleport_only"),
    default=str(_cfg_value(file_cfg, "env", "reset_mode", "isaac_reset")),
    help="Reset implementation used after episode termination. 'teleport_only' keeps the vehicle pool alive and teleports slots instead of running the heavy reset path.",
)
parser.add_argument(
    "--dynamics_mode",
    choices=("physx", "bicycle"),
    default=str(_cfg_value(file_cfg, "env", "dynamics_mode", "physx")),
    help="Vehicle dynamics backend. 'physx' uses full rigid-body articulation (default). 'bicycle' replaces PhysX with a kinematic bicycle model — useful for physics-gap ablations.",
)
parser.add_argument(
    "--test_mode",
    choices=(
        "none",
        "collision_test",
        "scene_factory_collision_test",
        "scene_factory_multiworld_random_steer_test",
        "scene_factory_policy_eval",
        "friction_ruler",
        "bicycle_sinwave_demo",
    ),
    default=str(_cfg_value(file_cfg, "test", "mode", "none")),
    help=(
        "Optional deterministic debug rollout mode. "
        "'collision_test' runs a flat-world head-on crash. "
        "'scene_factory_collision_test' runs the same head-on crash with SceneFactory roads enabled. "
        "'scene_factory_multiworld_random_steer_test' runs multiple SceneFactory worlds with full throttle and "
        "random steering. "
        "'scene_factory_policy_eval' loads a trained checkpoint and runs one deterministic episode per world."
    ),
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default=str(_cfg_value(file_cfg, "test", "checkpoint_path", "")),
    help="Checkpoint path used by scene_factory_policy_eval.",
)
parser.add_argument(
    "--eval_max_steps",
    type=int,
    default=int(_cfg_value(file_cfg, "test", "max_steps", 0)),
    help="Optional hard cap on evaluation steps. 0 uses one full episode horizon.",
)
parser.add_argument(
    "--double_time_allowance",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "test", "double_time_allowance", False)),
    help="For test modes, double the per-world episode horizon before timeout.",
)
parser.add_argument(
    "--invincible",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "test", "invincible", False)),
    help="If enabled, crash/collision/forbidden-lane events do not mark vehicles done or clear them out.",
)
parser.add_argument(
    "--random_od",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "test", "random_od", False)),
    help="If enabled, resample random origin-destination pairs on lane centerlines at each episode reset.",
)
parser.add_argument(
    "--random_od_min_travel_m",
    type=float,
    default=float(_cfg_value(file_cfg, "test", "random_od_min_travel_m", 20.0)),
    help="Minimum travel distance for randomly sampled OD pairs.",
)
parser.add_argument(
    "--random_od_max_travel_m",
    type=float,
    default=float(_cfg_value(file_cfg, "test", "random_od_max_travel_m", 60.0)),
    help="Maximum travel distance for randomly sampled OD pairs.",
)
parser.add_argument("--env_spacing", type=float, default=float(_cfg_value(file_cfg, "env", "env_spacing", 18.0)), help="Spacing between vectorized environments.")
parser.add_argument("--start_radius_m", type=float, default=float(_cfg_value(file_cfg, "env", "start_radius_m", 0.5)), help="Shared per-world spawn offset radius.")
parser.add_argument(
    "--agent_spawn_circle_radius_m",
    type=float,
    default=float(_cfg_value(file_cfg, "env", "agent_spawn_circle_radius_m", 3.5)),
    help="Radius of the within-world vehicle spawn ring.",
)
parser.add_argument(
    "--agent_spawn_jitter_m",
    type=float,
    default=float(_cfg_value(file_cfg, "env", "agent_spawn_jitter_m", 0.12)),
    help="Random XY jitter added to each vehicle spawn.",
)
parser.add_argument("--episode_length_s", type=float, default=float(_cfg_value(file_cfg, "env", "episode_length_s", 15.0)), help="Episode length in seconds.")
parser.add_argument("--goal_radius_min_m", type=float, default=float(_cfg_value(file_cfg, "env", "goal_radius_min_m", 5.0)), help="Minimum goal radius from env origin.")
parser.add_argument("--goal_radius_max_m", type=float, default=float(_cfg_value(file_cfg, "env", "goal_radius_max_m", 8.0)), help="Maximum goal radius from env origin.")
parser.add_argument(
    "--goal_reached_threshold_m",
    type=float,
    default=float(_cfg_value(file_cfg, "env", "goal_reached_threshold_m", 0.85)),
    help="Distance threshold for goal completion.",
)
parser.add_argument(
    "--fall_height_threshold_m",
    type=float,
    default=float(_cfg_value(file_cfg, "env", "fall_height_threshold_m", 0.18)),
    help="Crash threshold on root height relative to env origin.",
)
parser.add_argument(
    "--bad_tilt_gravity_threshold",
    type=float,
    default=float(_cfg_value(file_cfg, "env", "bad_tilt_gravity_threshold", -0.15)),
    help="Crash threshold on projected gravity z in body frame.",
)
parser.add_argument(
    "--max_distance_from_origin_m",
    type=float,
    default=float(_cfg_value(file_cfg, "env", "max_distance_from_origin_m", 14.0)),
    help="Logical world radius used for out-of-bounds termination.",
)
parser.add_argument(
    "--agent_neighbor_obs_scale_m",
    type=float,
    default=float(_cfg_value(file_cfg, "env", "agent_neighbor_obs_scale_m", 12.0)),
    help="Normalization scale for nearest-neighbor observation features.",
)
parser.add_argument(
    "--agent_collision_warmup_steps",
    type=int,
    default=int(_cfg_value(file_cfg, "env", "agent_collision_warmup_steps", 24)),
    help="Ignore inter-vehicle collision detection for this many environment steps after reset.",
)
parser.add_argument(
    "--replicate_physics",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "env", "replicate_physics", True)),
)
parser.add_argument(
    "--clone_in_fabric",
    choices=("auto", "true", "false"),
    default=str(_cfg_value(file_cfg, "env", "clone_in_fabric", "auto")),
)
parser.add_argument(
    "--rl_device",
    type=str,
    default=str(_cfg_value(file_cfg, "app", "rl_device", "")),
    help="Device for the PPO runner. Empty defaults to sim device.",
)
parser.add_argument(
    "--silence_runtime_warnings",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "app", "silence_runtime_warnings", False)),
    help="Suppress Isaac Kit warning-level runtime logs in the terminal for this run.",
)
parser.add_argument(
    "--use_fabric",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "sim", "use_fabric", True)),
    help="Enable Isaac Lab Fabric. Disable this for SceneFactory road stage-dump/debug runs to isolate PointInstancer issues.",
)
parser.add_argument("--num_steps_per_env", type=int, default=int(_cfg_value(file_cfg, "runner", "num_steps_per_env", 32)), help="Rollout steps per env for each PPO update.")
parser.add_argument("--save_interval", type=int, default=int(_cfg_value(file_cfg, "runner", "save_interval", 10)), help="Checkpoint save interval in iterations.")
parser.add_argument("--learning_rate", type=float, default=float(_cfg_value(file_cfg, "runner", "learning_rate", 3.0e-4)), help="PPO learning rate.")
parser.add_argument("--num_learning_epochs", type=int, default=int(_cfg_value(file_cfg, "runner", "num_learning_epochs", 5)), help="PPO epochs per update.")
parser.add_argument("--num_mini_batches", type=int, default=int(_cfg_value(file_cfg, "runner", "num_mini_batches", 4)), help="Number of PPO mini-batches per update.")
parser.add_argument("--entropy_coef", type=float, default=float(_cfg_value(file_cfg, "runner", "entropy_coef", 0.01)), help="PPO entropy coefficient.")
parser.add_argument("--clip_param", type=float, default=float(_cfg_value(file_cfg, "runner", "clip_param", 0.2)), help="PPO clip parameter.")
parser.add_argument("--desired_kl", type=float, default=float(_cfg_value(file_cfg, "runner", "desired_kl", 0.01)), help="Target KL for adaptive LR schedule.")
parser.add_argument("--gamma", type=float, default=float(_cfg_value(file_cfg, "runner", "gamma", 0.99)), help="PPO discount factor.")
parser.add_argument("--gae_lambda", type=float, default=float(_cfg_value(file_cfg, "runner", "gae_lambda", 0.95)), help="PPO GAE lambda.")
parser.add_argument("--experiment_name", type=str, default=str(_cfg_value(file_cfg, "runner", "experiment_name", "student_vehicle_goal_multiagent")), help="Experiment name.")
parser.add_argument("--run_name", type=str, default=str(_cfg_value(file_cfg, "runner", "run_name", "smoke")), help="Optional run-name suffix.")
parser.add_argument(
    "--shared_policy_mode",
    choices=("agent_slots", "joint_world"),
    default=str(_cfg_value(file_cfg, "runner", "shared_policy_mode", "agent_slots")),
    help=(
        "How to flatten the multi-agent env for PPO. "
        "'agent_slots' exposes one shared-policy slot per vehicle. "
        "'joint_world' keeps the older concatenated multi-agent world policy."
    ),
)
parser.add_argument(
    "--video",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "video", "enabled", False)),
    help="Record rollout videos during training using a fixed camera sensor.",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=int(_cfg_value(file_cfg, "video", "interval", 2500)),
    help="Global environment-step interval between recorded training videos.",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=int(_cfg_value(file_cfg, "video", "length", 300)),
    help="Number of environment steps captured in each training video.",
)
parser.add_argument(
    "--video_name_prefix",
    type=str,
    default=str(_cfg_value(file_cfg, "video", "name_prefix", "train")),
    help="Filename prefix for recorded training videos.",
)
parser.add_argument(
    "--video_width",
    type=int,
    default=int(_cfg_value(file_cfg, "video", "width", 1280)),
    help="Width in pixels for the fixed training capture camera.",
)
parser.add_argument(
    "--video_height",
    type=int,
    default=int(_cfg_value(file_cfg, "video", "height", 720)),
    help="Height in pixels for the fixed training capture camera.",
)
parser.add_argument(
    "--video_fps",
    type=int,
    default=int(_cfg_value(file_cfg, "video", "fps", 20)),
    help="Output fps for saved training clips.",
)
parser.add_argument(
    "--video_step_stride",
    type=int,
    default=int(_cfg_value(file_cfg, "video", "step_stride", 1)),
    help="Capture one video frame every N environment steps while recording.",
)
parser.add_argument(
    "--video_view_mode",
    choices=("whole_grid", "single_env", "per_env"),
    default=str(_cfg_value(file_cfg, "video", "view_mode", "whole_grid")),
    help="Capture either the whole training grid, a single environment, or one video per environment.",
)
parser.add_argument(
    "--video_env_index",
    type=int,
    default=int(_cfg_value(file_cfg, "video", "env_index", 0)),
    help="Environment index to focus when video_view_mode=single_env.",
)
parser.add_argument(
    "--video_vehicle_proxy_markers",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "video", "vehicle_proxy_markers", False)),
    help="Draw color-coded root-pose proxy boxes over vehicles in debug/video renders.",
)
parser.add_argument(
    "--video_vehicle_proxy_z_offset_m",
    type=float,
    default=float(_cfg_value(file_cfg, "video", "vehicle_proxy_z_offset_m", 0.0)),
    help="Vertical offset for the vehicle proxy boxes in debug/video renders.",
)
parser.add_argument(
    "--video_camera_pose_mode",
    type=str,
    default=str(_cfg_value(file_cfg, "video", "camera_pose_mode", "top_down")),
    choices=["top_down", "traffic_cam", "flyover", "flyover_drift"],
    help="Camera pose mode: top_down (bird's-eye), traffic_cam (tilted, low), or flyover (cinematic rise).",
)
parser.add_argument(
    "--video_traffic_cam_height_m",
    type=float,
    default=float(_cfg_value(file_cfg, "video", "traffic_cam_height_m", 7.0)),
    help="Traffic-cam eye height in meters.",
)
parser.add_argument(
    "--video_traffic_cam_distance_m",
    type=float,
    default=float(_cfg_value(file_cfg, "video", "traffic_cam_distance_m", 25.0)),
    help="Traffic-cam horizontal offset behind scene center.",
)
parser.add_argument(
    "--video_traffic_cam_look_height_m",
    type=float,
    default=float(_cfg_value(file_cfg, "video", "traffic_cam_look_height_m", 0.5)),
    help="Height the traffic-cam looks at.",
)
parser.add_argument(
    "--video_traffic_cam_azimuth_deg",
    type=float,
    default=float(_cfg_value(file_cfg, "video", "traffic_cam_azimuth_deg", 0.0)),
    help="Azimuth rotation in degrees. 0=camera south of scene looking north. Positive=rotate left (CCW from above).",
)
parser.add_argument(
    "--video_traffic_cam_lateral_offset_m",
    type=float,
    default=0.0,
    help="Shift camera left (positive) or right (negative) in meters.",
)
# Flyover camera args
parser.add_argument("--video_flyover_start_height_m", type=float, default=8.0, help="Flyover: starting camera height.")
parser.add_argument("--video_flyover_end_height_m", type=float, default=200.0, help="Flyover: final camera height.")
parser.add_argument("--video_flyover_surveillance_frames", type=int, default=120, help="Flyover phase 1: hold as surveillance cam.")
parser.add_argument("--video_flyover_tilt_frames", type=int, default=180, help="Flyover phase 2: tilt up to reveal neighbors.")
parser.add_argument("--video_flyover_zoomout_frames", type=int, default=300, help="Flyover phase 3: gentle rise.")
parser.add_argument("--video_flyover_start_env_index", type=int, default=0, help="Flyover: env index to stay centered on.")
parser.add_argument("--video_flyover_start_tilt_deg", type=float, default=25.0, help="Flyover: low surveillance angle from horizontal.")
parser.add_argument("--video_flyover_end_tilt_deg", type=float, default=75.0, help="Flyover: near top-down angle at end.")
parser.add_argument("--video_flyover_start_distance_m", type=float, default=25.0, help="Flyover: horizontal offset at start.")
parser.add_argument("--video_flyover_lookaway_frames", type=int, default=0, help="Flyover phase 4: tilt head up + pan left (0=disabled).")
parser.add_argument("--video_flyover_lookaway_pitch_deg", type=float, default=45.0, help="Flyover phase 4: degrees to tilt up.")
parser.add_argument("--video_flyover_lookaway_yaw_deg", type=float, default=45.0, help="Flyover phase 4: degrees to pan left.")
# Flyover-drift camera args
parser.add_argument("--video_drift_rise_frames", type=int, default=300, help="Drift: frames to rise to overview.")
parser.add_argument("--video_drift_lateral_frames", type=int, default=300, help="Drift: frames to drift left.")
parser.add_argument("--video_drift_pan_frames", type=int, default=300, help="Drift: frames to pan right + pitch up.")
parser.add_argument("--video_drift_hold_frames", type=int, default=0, help="Drift: frames to hold final pose.")
parser.add_argument("--video_drift_start_height_m", type=float, default=8.0, help="Drift: starting camera height.")
parser.add_argument("--video_drift_rise_height_m", type=float, default=400.0, help="Drift: height at end of rise.")
parser.add_argument("--video_drift_lateral_distance_m", type=float, default=600.0, help="Drift: how far to drift left.")
parser.add_argument("--video_drift_pan_yaw_deg", type=float, default=90.0, help="Drift: how far to pan right.")
parser.add_argument("--video_drift_pitch_up_deg", type=float, default=30.0, help="Drift: how far to pitch up at end.")
parser.add_argument("--video_drift_start_tilt_deg", type=float, default=25.0, help="Drift: initial surveillance tilt.")
parser.add_argument("--video_drift_rise_tilt_deg", type=float, default=70.0, help="Drift: tilt at top of rise.")
parser.add_argument("--video_drift_azimuth_deg", type=float, default=0.0, help="Drift: initial viewing direction.")
parser.add_argument("--video_flyover_azimuth_deg", type=float, default=0.0, help="Flyover: viewing direction.")
parser.add_argument(
    "--road_hidden_types",
    type=str,
    default=str(_cfg_value(file_cfg, "road", "hidden_types", "")),
    help="Comma-separated road type ints to hide visually (e.g. '1,2,3' for lane centers). Still in scene for obs/physics.",
)
parser.add_argument(
    "--hide_goal_markers",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "video", "hide_goal_markers", False)),
    help="Hide destination beacon visuals for clean video capture.",
)
parser.add_argument(
    "--resume_from",
    type=str,
    default=str(_cfg_value(file_cfg, "runner", "resume_path", "")),
    help="Path to a model checkpoint (.pt) to resume training from. Loads weights + optimizer state.",
)
parser.add_argument(
    "--save_stage_usd",
    type=str,
    default=str(_cfg_value(file_cfg, "debug", "save_stage_usd", "")),
    help="Optional path to export the initialized training stage before learning starts.",
)
parser.add_argument(
    "--exit_after_stage_save",
    action=argparse.BooleanOptionalAction,
    default=bool(_cfg_value(file_cfg, "debug", "exit_after_stage_save", False)),
    help="Exit immediately after saving the initialized stage debug dump.",
)
parser.add_argument(
    "--fixed_action",
    type=float,
    nargs=3,
    default=None,
    metavar=("THROTTLE", "STEER", "BRAKE"),
    help="Override policy actions with fixed [throttle, steer, brake] each in [-1,1]. "
         "Useful for friction demo videos where all vehicles get identical commands.",
)
parser.add_argument(
    "--action_schedule",
    type=str,
    nargs="+",
    default=None,
    metavar="STEP:T,S,B",
    help="Time-varying scripted actions. Each entry is step:throttle,steer,brake. "
         "E.g. --action_schedule 0:1.0,0.0,0.0 60:1.0,1.0,0.0  means full throttle "
         "straight for 60 steps then full throttle + full steer. Overrides --fixed_action.",
)
AppLauncher.add_app_launcher_args(parser)
_device_from_cfg = _cfg_value(file_cfg, "app", "device", None)
parser.set_defaults(
    device=str(_device_from_cfg) if _device_from_cfg else "cuda:0",
    enable_cameras=bool(_cfg_value(file_cfg, "video", "enabled", False)),
)
args_cli = parser.parse_args()


def _configure_headless_camera_environment(args: argparse.Namespace) -> None:
    """Force a true offscreen path for headless camera/video runs.

    On workstation setups with an active X display, Isaac Sim can still attempt a
    GLX-backed initialization even when `--headless` is set. That breaks video
    capture with errors such as `GLXBadFBConfig`. For headless runs that need
    cameras, scrub GUI display variables before AppLauncher starts the app.
    """

    needs_offscreen_cameras = bool(getattr(args, "headless", False)) and bool(
        getattr(args, "video", False) or getattr(args, "enable_cameras", False)
    )
    if not needs_offscreen_cameras:
        return

    if os.environ.get("DISPLAY"):
        print(
            f"[INFO][SceneFactory]: Unsetting DISPLAY={os.environ['DISPLAY']} for headless camera/video rendering."
        )
        os.environ.pop("DISPLAY", None)
    os.environ.setdefault("HEADLESS", "1")
    os.environ.setdefault("ENABLE_CAMERAS", "1")


_configure_headless_camera_environment(args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _configure_runtime_warning_filter() -> None:
    if not bool(getattr(args_cli, "silence_runtime_warnings", False)):
        return
    try:
        import carb.settings

        carb_settings = carb.settings.get_settings()
        carb_settings.set_string("/log/outputStreamLevel", "Error")
        print("[INFO][SceneFactory] Suppressing Kit warning-level runtime logs for this run.")
    except Exception as exc:
        print(f"[WARN][SceneFactory] Failed to configure runtime warning filter: {exc}")


_configure_runtime_warning_filter()


import torch
import gymnasium as gym
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
)

from rsl_rl.runners import on_policy_runner as rsl_on_policy_runner_module

from src.scene_factory_late_fusion_actor_critic import SceneFactoryLateFusionActorCritic
from src.student_vehicle_goal_env import DEFAULT_STUDENT_VEHICLE_USD
from src.student_vehicle_multiagent_goal_env import (
    StudentVehicleMultiAgentGoalEnv,
    StudentVehicleMultiAgentGoalEnvCfg,
    _reference_road_point_feat_dim,
    _reference_vehicle_feat_dim,
    configure_multi_agent_spaces,
)
from src.trfc import weather_context_dim


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _register_scene_factory_custom_policy_classes() -> None:
    rsl_on_policy_runner_module.SceneFactoryLateFusionActorCritic = SceneFactoryLateFusionActorCritic


def _resolve_seed(seed: int) -> int:
    if int(seed) >= 0:
        return int(seed)
    return random.randint(0, 10_000)


def _build_env_cfg() -> StudentVehicleMultiAgentGoalEnvCfg:
    cfg = StudentVehicleMultiAgentGoalEnvCfg()
    cfg.seed = _resolve_seed(args_cli.seed)
    cfg.scene.num_envs = int(args_cli.num_envs)
    cfg.scene.env_spacing = float(args_cli.env_spacing)
    cfg.scene.replicate_physics = bool(args_cli.replicate_physics)
    if args_cli.device is not None:
        cfg.sim.device = str(args_cli.device)
    else:
        cfg.sim.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.sim.use_fabric = bool(args_cli.use_fabric)
    use_gpu_device = not str(cfg.sim.device).lower().startswith("cpu")
    requires_camera_rendering = bool(args_cli.video) or bool(getattr(args_cli, "enable_cameras", False))
    if args_cli.clone_in_fabric == "auto":
        cfg.scene.clone_in_fabric = (
            bool(getattr(args_cli, "headless", False))
            and cfg.scene.replicate_physics
            and use_gpu_device
            and not requires_camera_rendering
        )
    else:
        cfg.scene.clone_in_fabric = (
            args_cli.clone_in_fabric == "true" and cfg.scene.replicate_physics and use_gpu_device
        )
    if cfg.scene.clone_in_fabric and requires_camera_rendering:
        print("[INFO][SceneFactory]: Disabling Fabric cloning for camera/video capture runs.")
        cfg.scene.clone_in_fabric = False
    grid_extent = max(1.0, math.ceil(math.sqrt(max(1, cfg.scene.num_envs))) * float(cfg.scene.env_spacing))
    world_extent = max(
        float(args_cli.max_distance_from_origin_m),
        float(args_cli.goal_radius_max_m),
        float(args_cli.agent_spawn_circle_radius_m) + 10.0,
    )
    viewer_extent = max(grid_extent, 1.25 * world_extent)
    cfg.viewer.eye = (max(20.0, viewer_extent), max(20.0, viewer_extent), max(16.0, 0.8 * viewer_extent))
    cfg.viewer.lookat = (0.0, 0.0, 0.0)
    cfg.spawn_height_m = float(args_cli.spawn_height_m)
    cfg.spawn_yaw_noise_rad = float(args_cli.spawn_yaw_noise_rad)
    cfg.ground_mode = str(args_cli.ground_mode)
    cfg.use_scene_factory_roads = bool(args_cli.use_scene_factory_roads)
    cfg.scene_factory_config_path = str(Path(args_cli.scene_factory_config).expanduser().resolve())
    cfg.scene_factory_world_index = int(args_cli.scene_factory_world_index)
    cfg.scene_factory_world_selection_mode = str(args_cli.scene_factory_world_selection_mode)
    cfg.scene_factory_random_world_seed = int(args_cli.scene_factory_random_world_seed)
    cfg.reset_mode = str(args_cli.reset_mode)
    if bool(cfg.use_scene_factory_roads):
        scene_factory_cfg = _load_yaml_config(cfg.scene_factory_config_path)
        road_cfg = dict(scene_factory_cfg.get("road", {}) or {})
        vehicles_cfg = dict(scene_factory_cfg.get("vehicles", {}) or {})
        if "max_controllable_per_world" in vehicles_cfg:
            print(
                "[INFO][SceneFactory]: env.num_agents_per_env from the training preset governs runtime controllable "
                f"agents per scene ({int(args_cli.num_agents_per_env)} requested); "
                "vehicles.max_controllable_per_world in the scene config is not used as a training-time override."
            )
        if str(road_cfg.get("render_mode", "point_instancer")).strip().lower() == "point_instancer":
            print(
                "[INFO][SceneFactory]: road_render_mode=point_instancer "
                f"use_fabric={cfg.sim.use_fabric} clone_in_fabric={cfg.scene.clone_in_fabric}"
            )
        if str(cfg.reset_mode).strip().lower().replace("-", "_") == "teleport_only":
            print("[INFO][SceneFactory]: reset_mode=teleport_only requested from the training preset.")
    cfg.start_radius_m = float(args_cli.start_radius_m)
    cfg.decimation = int(args_cli.decimation)
    cfg.sim.render_interval = int(args_cli.decimation)
    cfg.apply_runtime_external_wrench = bool(args_cli.apply_runtime_external_wrench)
    cfg.agent_spawn_circle_radius_m = float(args_cli.agent_spawn_circle_radius_m)
    cfg.agent_spawn_jitter_m = float(args_cli.agent_spawn_jitter_m)
    cfg.episode_length_s = float(args_cli.episode_length_s)
    cfg.goal_radius_min_m = float(args_cli.goal_radius_min_m)
    cfg.goal_radius_max_m = float(args_cli.goal_radius_max_m)
    cfg.goal_reached_threshold_m = float(args_cli.goal_reached_threshold_m)
    cfg.fall_height_threshold_m = float(args_cli.fall_height_threshold_m)
    cfg.bad_tilt_gravity_threshold = float(args_cli.bad_tilt_gravity_threshold)
    cfg.max_distance_from_origin_m = float(args_cli.max_distance_from_origin_m)
    cfg.agent_neighbor_obs_scale_m = float(args_cli.agent_neighbor_obs_scale_m)
    cfg.agent_collision_warmup_steps = int(args_cli.agent_collision_warmup_steps)
    cfg.observation_mode = str(args_cli.observation_mode)
    cfg.obs_weather_context_enable = bool(args_cli.obs_weather_context_enable)
    cfg.obs_weather_context_blind = bool(args_cli.obs_weather_context_blind)
    cfg.obs_road_points_enable = bool(args_cli.obs_road_points_enable)
    cfg.obs_road_points_k = int(args_cli.obs_road_points_k)
    cfg.obs_road_points_radius_m = float(args_cli.obs_road_points_radius_m)
    cfg.obs_road_points_type_norm = float(args_cli.obs_road_points_type_norm)
    cfg.obs_road_points_mode = str(args_cli.obs_road_points_mode)
    cfg.obs_road_points_include_dirs = bool(args_cli.obs_road_points_include_dirs)
    cfg.obs_neighbor_enable = bool(args_cli.obs_neighbor_enable)
    cfg.obs_neighbor_k = int(args_cli.obs_neighbor_k)
    cfg.obs_neighbor_include_ttc = bool(args_cli.obs_neighbor_include_ttc)
    cfg.obs_neighbor_include_index = bool(args_cli.obs_neighbor_include_index)
    cfg.obs_neighbor_ttc_max_s = float(args_cli.obs_neighbor_ttc_max_s)
    cfg.obs_timing_print_enable = bool(args_cli.obs_timing_print_enable)
    cfg.obs_timing_print_every_n = int(args_cli.obs_timing_print_every_n)
    cfg.step_timing_log_enable = bool(args_cli.step_timing_log_enable)
    cfg.step_timing_print_enable = bool(args_cli.step_timing_print_enable)
    cfg.step_timing_print_every_n = int(args_cli.step_timing_print_every_n)
    cfg.step_timing_cuda_sync_enable = bool(args_cli.step_timing_cuda_sync_enable)
    cfg.reward_lane_center_enable = bool(args_cli.reward_lane_center_enable)
    cfg.reward_lane_center_types = _cfg_int_tuple(file_cfg, "reward", "lane_center_types", (1, 2))
    cfg.reward_lane_center_per_step = float(args_cli.reward_lane_center_per_step)
    cfg.reward_lane_forbidden_enable = bool(args_cli.reward_lane_forbidden_enable)
    cfg.reward_lane_forbidden_types = _cfg_int_tuple(file_cfg, "reward", "lane_forbidden_types", (15, 16))
    cfg.reward_lane_forbidden_penalty = float(args_cli.reward_lane_forbidden_penalty)
    cfg.reward_collision_penalty = float(args_cli.reward_collision_penalty)
    cfg.reward_crash_penalty = float(args_cli.reward_crash_penalty)
    cfg.reward_mode = str(args_cli.reward_mode)
    cfg.reward_goal_bonus = float(args_cli.reward_goal_bonus)
    cfg.reward_choco_offroad_penalty = float(args_cli.reward_choco_offroad_penalty)
    cfg.reward_choco_idle_penalty_enable = bool(args_cli.reward_choco_idle_penalty_enable)
    cfg.reward_choco_idle_penalty_per_step = float(args_cli.reward_choco_idle_penalty_per_step)
    cfg.reward_choco_idle_speed_threshold_mps = float(args_cli.reward_choco_idle_speed_threshold_mps)
    cfg.reward_choco_speed_bonus_enable = bool(args_cli.reward_choco_speed_bonus_enable)
    cfg.reward_choco_speed_bonus_per_step = float(args_cli.reward_choco_speed_bonus_per_step)
    cfg.reward_choco_speed_bonus_max_mps = float(args_cli.reward_choco_speed_bonus_max_mps)
    cfg.reward_choco_geom_lane_enable = bool(args_cli.reward_choco_geom_lane_enable)
    cfg.reward_choco_geom_lane_per_step = float(args_cli.reward_choco_geom_lane_per_step)
    cfg.reward_choco_geom_lane_tolerance_m = float(args_cli.reward_choco_geom_lane_tolerance_m)
    cfg.reward_choco_geom_lane_heading_weight = float(args_cli.reward_choco_geom_lane_heading_weight)
    cfg.reward_choco_geom_lane_min_alignment = float(args_cli.reward_choco_geom_lane_min_alignment)
    cfg.reward_choco_geom_route_progress_weight = float(args_cli.reward_choco_geom_route_progress_weight)
    cfg.reward_choco_geom_offroad_enable = bool(args_cli.reward_choco_geom_offroad_enable)
    cfg.reward_choco_geom_offroad_lateral_threshold_m = float(args_cli.reward_choco_geom_offroad_lateral_threshold_m)
    cfg.reward_choco_geom_offroad_distance_threshold_m = float(args_cli.reward_choco_geom_offroad_distance_threshold_m)
    cfg.reward_choco_ttc_penalty_enable = bool(args_cli.reward_choco_ttc_penalty_enable)
    cfg.reward_choco_ttc_penalty_alpha = float(args_cli.reward_choco_ttc_penalty_alpha)
    cfg.reward_choco_ttc_penalty_max = float(args_cli.reward_choco_ttc_penalty_max)
    cfg.reward_choco_ttc_penalty_min_ttc = float(args_cli.reward_choco_ttc_penalty_min_ttc)
    cfg.reward_choco_road_edge_ttc_penalty_enable = bool(args_cli.reward_choco_road_edge_ttc_penalty_enable)
    cfg.reward_choco_road_edge_ttc_penalty_alpha = float(args_cli.reward_choco_road_edge_ttc_penalty_alpha)
    cfg.reward_choco_road_edge_ttc_penalty_max = float(args_cli.reward_choco_road_edge_ttc_penalty_max)
    cfg.reward_choco_road_edge_ttc_penalty_min_ttc = float(args_cli.reward_choco_road_edge_ttc_penalty_min_ttc)
    cfg.reward_choco_road_edge_ttc_hard_min_ttc = float(args_cli.reward_choco_road_edge_ttc_hard_min_ttc)
    cfg.reward_choco_road_edge_ttc_radius_m = float(args_cli.reward_choco_road_edge_ttc_radius_m)
    cfg.test_mode = str(args_cli.test_mode).strip().lower()
    cfg.invincible = bool(args_cli.invincible)
    cfg.random_od = bool(args_cli.random_od)
    cfg.random_od_min_travel_m = float(args_cli.random_od_min_travel_m)
    cfg.random_od_max_travel_m = float(args_cli.random_od_max_travel_m)
    if bool(args_cli.double_time_allowance) and cfg.test_mode != "none":
        cfg.episode_length_s = float(cfg.episode_length_s) * 2.0
        print(
            "[INFO][SceneFactory] double_time_allowance=true: "
            f"episode_length_s doubled to {cfg.episode_length_s:.2f}s for test mode {cfg.test_mode}.",
            flush=True,
        )
    cfg.collision_test_post_collision_steps = int(_cfg_value(file_cfg, "test", "post_collision_steps", 120))
    cfg.collision_test_post_collision_throttle = float(
        _cfg_value(file_cfg, "test", "post_collision_throttle", 0.0)
    )
    cfg.collision_test_post_collision_steering = float(
        _cfg_value(file_cfg, "test", "post_collision_steering", 0.0)
    )
    cfg.collision_test_post_collision_brake = float(_cfg_value(file_cfg, "test", "post_collision_brake", 1.0))
    cfg.random_steer_test_settle_steps = int(_cfg_value(file_cfg, "test", "settle_steps", 24))
    cfg.random_steer_test_drive_steps = int(_cfg_value(file_cfg, "test", "drive_steps", 600))
    cfg.random_steer_test_throttle = float(_cfg_value(file_cfg, "test", "throttle", 1.0))
    cfg.random_steer_test_brake = float(_cfg_value(file_cfg, "test", "brake", 0.0))
    cfg.random_steer_test_steering_min = float(_cfg_value(file_cfg, "test", "steering_min", -1.0))
    cfg.random_steer_test_steering_max = float(_cfg_value(file_cfg, "test", "steering_max", 1.0))
    cfg.random_steer_test_steering_hold_steps = int(_cfg_value(file_cfg, "test", "steering_hold_steps", 12))
    cfg.random_steer_test_seed = int(_cfg_value(file_cfg, "test", "seed", 123))
    cfg.capture_camera_enabled = bool(args_cli.video)
    cfg.capture_camera_width = int(args_cli.video_width)
    cfg.capture_camera_height = int(args_cli.video_height)
    cfg.capture_camera_view_mode = str(args_cli.video_view_mode)
    cfg.capture_camera_env_index = int(args_cli.video_env_index)
    cfg.capture_camera_pose_mode = str(args_cli.video_camera_pose_mode)
    cfg.capture_camera_traffic_cam_height_m = float(args_cli.video_traffic_cam_height_m)
    cfg.capture_camera_traffic_cam_distance_m = float(args_cli.video_traffic_cam_distance_m)
    cfg.capture_camera_traffic_cam_look_height_m = float(args_cli.video_traffic_cam_look_height_m)
    cfg.capture_camera_traffic_cam_azimuth_deg = float(args_cli.video_traffic_cam_azimuth_deg)
    cfg.capture_camera_traffic_cam_lateral_offset_m = float(args_cli.video_traffic_cam_lateral_offset_m)
    cfg.capture_camera_traffic_cam_lateral_offset_m = float(args_cli.video_traffic_cam_lateral_offset_m)
    # Flyover
    cfg.capture_camera_flyover_start_height_m = float(args_cli.video_flyover_start_height_m)
    cfg.capture_camera_flyover_end_height_m = float(args_cli.video_flyover_end_height_m)
    cfg.capture_camera_flyover_surveillance_frames = int(args_cli.video_flyover_surveillance_frames)
    cfg.capture_camera_flyover_tilt_frames = int(args_cli.video_flyover_tilt_frames)
    cfg.capture_camera_flyover_zoomout_frames = int(args_cli.video_flyover_zoomout_frames)
    cfg.capture_camera_flyover_start_env_index = int(args_cli.video_flyover_start_env_index)
    cfg.capture_camera_flyover_start_tilt_deg = float(args_cli.video_flyover_start_tilt_deg)
    cfg.capture_camera_flyover_end_tilt_deg = float(args_cli.video_flyover_end_tilt_deg)
    cfg.capture_camera_flyover_start_distance_m = float(args_cli.video_flyover_start_distance_m)
    cfg.capture_camera_flyover_azimuth_deg = float(args_cli.video_flyover_azimuth_deg)
    cfg.capture_camera_flyover_lookaway_frames = int(args_cli.video_flyover_lookaway_frames)
    cfg.capture_camera_flyover_lookaway_pitch_deg = float(args_cli.video_flyover_lookaway_pitch_deg)
    cfg.capture_camera_flyover_lookaway_yaw_deg = float(args_cli.video_flyover_lookaway_yaw_deg)
    # Flyover-drift
    cfg.capture_camera_drift_rise_frames = int(args_cli.video_drift_rise_frames)
    cfg.capture_camera_drift_lateral_frames = int(args_cli.video_drift_lateral_frames)
    cfg.capture_camera_drift_pan_frames = int(args_cli.video_drift_pan_frames)
    cfg.capture_camera_drift_hold_frames = int(args_cli.video_drift_hold_frames)
    cfg.capture_camera_drift_start_height_m = float(args_cli.video_drift_start_height_m)
    cfg.capture_camera_drift_rise_height_m = float(args_cli.video_drift_rise_height_m)
    cfg.capture_camera_drift_lateral_distance_m = float(args_cli.video_drift_lateral_distance_m)
    cfg.capture_camera_drift_pan_yaw_deg = float(args_cli.video_drift_pan_yaw_deg)
    cfg.capture_camera_drift_pitch_up_deg = float(args_cli.video_drift_pitch_up_deg)
    cfg.capture_camera_drift_start_tilt_deg = float(args_cli.video_drift_start_tilt_deg)
    cfg.capture_camera_drift_rise_tilt_deg = float(args_cli.video_drift_rise_tilt_deg)
    cfg.capture_camera_drift_azimuth_deg = float(args_cli.video_drift_azimuth_deg)
    cfg.vehicle_proxy_marker_enable = bool(args_cli.video_vehicle_proxy_markers)
    cfg.vehicle_proxy_marker_z_offset_m = float(args_cli.video_vehicle_proxy_z_offset_m)
    # Road type visual hiding (still in scene for obs/physics)
    _rht = str(args_cli.road_hidden_types).strip()
    cfg.road_hidden_types = [int(x) for x in _rht.split(",") if x.strip()] if _rht else None
    cfg.hide_goal_markers = bool(args_cli.hide_goal_markers)
    cfg.student_usd_path = str(Path(args_cli.student_usd or DEFAULT_STUDENT_VEHICLE_USD).expanduser().resolve())
    if str(args_cli.tunable_config_json):
        cfg.tunable_config_json = str(Path(args_cli.tunable_config_json).expanduser().resolve())
    if cfg.test_mode == "collision_test":
        if cfg.scene.num_envs != 1:
            print("[INFO][SceneFactory] collision_test forces num_envs=1.")
        cfg.scene.num_envs = 1
        if int(args_cli.num_agents_per_env) != 2:
            print("[INFO][SceneFactory] collision_test forces num_agents_per_env=2.")
        args_cli.num_agents_per_env = 2
        if cfg.use_scene_factory_roads:
            print(
                "[INFO][SceneFactory] collision_test disables SceneFactory roads and uses a flat plane "
                "to isolate vehicle-vehicle contact and rendering."
            )
        cfg.use_scene_factory_roads = False
        if bool(args_cli.video) and not bool(args_cli.use_fabric):
            print("[INFO][SceneFactory] collision_test enables Fabric for headless vehicle video capture.")
        cfg.sim.use_fabric = True if bool(args_cli.video) else bool(args_cli.use_fabric)
    elif cfg.test_mode == "scene_factory_collision_test":
        if cfg.scene.num_envs != 1:
            print("[INFO][SceneFactory] scene_factory_collision_test forces num_envs=1.")
        cfg.scene.num_envs = 1
        if int(args_cli.num_agents_per_env) != 2:
            print("[INFO][SceneFactory] scene_factory_collision_test forces num_agents_per_env=2.")
        args_cli.num_agents_per_env = 2
        if not cfg.use_scene_factory_roads:
            print("[INFO][SceneFactory] scene_factory_collision_test enables SceneFactory roads.")
        cfg.use_scene_factory_roads = True
        if bool(args_cli.video) and not bool(args_cli.use_fabric):
            print("[INFO][SceneFactory] scene_factory_collision_test enables Fabric for headless vehicle video capture.")
        cfg.sim.use_fabric = True if bool(args_cli.video) else bool(args_cli.use_fabric)
    elif cfg.test_mode == "scene_factory_multiworld_random_steer_test":
        if not cfg.use_scene_factory_roads:
            print("[INFO][SceneFactory] scene_factory_multiworld_random_steer_test enables SceneFactory roads.")
        cfg.use_scene_factory_roads = True
        if bool(args_cli.video) and not bool(args_cli.use_fabric):
            print(
                "[INFO][SceneFactory] scene_factory_multiworld_random_steer_test enables Fabric "
                "for headless vehicle video capture."
            )
        cfg.sim.use_fabric = True if bool(args_cli.video) else bool(args_cli.use_fabric)
    elif cfg.test_mode == "scene_factory_policy_eval":
        if not cfg.use_scene_factory_roads:
            print("[INFO][SceneFactory] scene_factory_policy_eval enables SceneFactory roads.")
        cfg.use_scene_factory_roads = True
    elif cfg.test_mode == "bicycle_sinwave_demo":
        if not cfg.use_scene_factory_roads:
            print("[INFO][SceneFactory] bicycle_sinwave_demo enables SceneFactory roads.")
        cfg.use_scene_factory_roads = True
        cfg.dynamics_mode = "bicycle"
        cfg.invincible = True  # don't terminate on tilt/fall -- bicycle has no roll physics
        if bool(args_cli.video) and not bool(args_cli.use_fabric):
            print("[INFO][SceneFactory] bicycle_sinwave_demo enables Fabric for headless video capture.")
        cfg.sim.use_fabric = True if bool(args_cli.video) else bool(args_cli.use_fabric)
        print("[INFO][SceneFactory] bicycle_sinwave_demo: dynamics_mode=bicycle, invincible=True", flush=True)
    elif cfg.test_mode == "friction_ruler":
        cfg.use_scene_factory_roads = False
        cfg.friction_ruler_mode = True
        cfg.friction_ruler_mu_values = str(_cfg_value(file_cfg, "env", "friction_ruler_mu_values", "1.1,0.6,0.3,0.1"))
        cfg.friction_ruler_labels = str(_cfg_value(file_cfg, "env", "friction_ruler_labels", ""))
        print(
            f"[INFO][FrictionRuler] friction_ruler mode: roads disabled, "
            f"mu_values={cfg.friction_ruler_mu_values}",
            flush=True,
        )
    configure_multi_agent_spaces(cfg, int(args_cli.num_agents_per_env))

    # ── Dynamics mode (bicycle overrides physx for any run mode) ──
    if str(args_cli.dynamics_mode) == "bicycle" and cfg.test_mode != "bicycle_sinwave_demo":
        cfg.dynamics_mode = "bicycle"
        cfg.invincible = True  # no roll/flip physics in bicycle mode
        print("[INFO][SceneFactory] dynamics_mode=bicycle: kinematic bicycle model active, invincible=True", flush=True)

    # ── Scripted action overrides (from config YAML) ──
    cfg.fixed_action = str(_cfg_value(file_cfg, "env", "fixed_action", ""))
    cfg.action_schedule = str(_cfg_value(file_cfg, "env", "action_schedule", ""))

    return cfg


def _build_runner_cfg(sim_device: str) -> RslRlOnPolicyRunnerCfg:
    rl_device = str(args_cli.rl_device or sim_device)
    policy_type = str(_cfg_value(file_cfg, "policy", "type", "mlp")).strip().lower().replace("-", "_")
    policy_class_name = "SceneFactoryLateFusionActorCritic" if policy_type == "late_fusion" else "ActorCritic"
    return RslRlOnPolicyRunnerCfg(
        seed=int(_resolve_seed(args_cli.seed)),
        device=rl_device,
        num_steps_per_env=int(args_cli.num_steps_per_env),
        max_iterations=int(args_cli.max_iterations),
        save_interval=max(1, int(args_cli.save_interval)),
        experiment_name=str(args_cli.experiment_name),
        run_name=str(args_cli.run_name),
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        clip_actions=1.0,
        logger="tensorboard",
        policy=RslRlPpoActorCriticCfg(
            class_name=policy_class_name,
            init_noise_std=1.0,
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            actor_hidden_dims=[256, 256],
            critic_hidden_dims=[256, 256],
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=float(args_cli.clip_param),
            entropy_coef=float(args_cli.entropy_coef),
            num_learning_epochs=int(args_cli.num_learning_epochs),
            num_mini_batches=int(args_cli.num_mini_batches),
            learning_rate=float(args_cli.learning_rate),
            schedule="adaptive",
            gamma=float(args_cli.gamma),
            lam=float(args_cli.gae_lambda),
            desired_kl=float(args_cli.desired_kl),
            max_grad_norm=1.0,
        ),
    )


def _build_late_fusion_policy_kwargs(env_cfg: StudentVehicleMultiAgentGoalEnvCfg) -> dict[str, Any]:
    ego_dim = 7 + (int(weather_context_dim()) if bool(env_cfg.obs_weather_context_enable) else 0)
    road_point_dim = (
        int(_reference_road_point_feat_dim(env_cfg.obs_road_points_include_dirs)) if bool(env_cfg.obs_road_points_enable) else 0
    )
    road_point_k = int(env_cfg.obs_road_points_k) if bool(env_cfg.obs_road_points_enable) else 0
    vehicle_dim = (
        int(_reference_vehicle_feat_dim(env_cfg.obs_neighbor_include_ttc, env_cfg.obs_neighbor_include_index))
        if bool(env_cfg.obs_neighbor_enable)
        else 0
    )
    vehicle_k = int(env_cfg.obs_neighbor_k) if bool(env_cfg.obs_neighbor_enable) else 0
    return {
        "ego_dim": int(_cfg_value(file_cfg, "policy", "ego_dim", ego_dim)),
        "road_point_dim": int(_cfg_value(file_cfg, "policy", "road_point_dim", road_point_dim)),
        "road_point_k": int(_cfg_value(file_cfg, "policy", "road_point_k", road_point_k)),
        "vehicle_dim": int(_cfg_value(file_cfg, "policy", "vehicle_dim", vehicle_dim)),
        "vehicle_k": int(_cfg_value(file_cfg, "policy", "vehicle_k", vehicle_k)),
        "ego_layers": list(_cfg_value(file_cfg, "policy", "ego_layers", [64, 64])),
        "road_layers": list(_cfg_value(file_cfg, "policy", "road_layers", [96, 96])),
        "vehicle_layers": list(_cfg_value(file_cfg, "policy", "vehicle_layers", [96, 96])),
        "shared_layers": list(_cfg_value(file_cfg, "policy", "shared_layers", [128, 64])),
        "last_layer_dim_pi": int(_cfg_value(file_cfg, "policy", "last_layer_dim_pi", 64)),
        "last_layer_dim_vf": int(_cfg_value(file_cfg, "policy", "last_layer_dim_vf", 64)),
        "activation": str(_cfg_value(file_cfg, "policy", "activation", "relu")),
        "dropout": float(_cfg_value(file_cfg, "policy", "dropout", 0.0)),
        "pool": str(_cfg_value(file_cfg, "policy", "pool", "max")),
    }


def _make_run_dir(log_root: Path, runner_cfg: RslRlOnPolicyRunnerCfg) -> Path:
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if runner_cfg.run_name:
        run_name += f"_{runner_cfg.run_name}"
    run_dir = log_root / runner_cfg.experiment_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_resolved_config(
    env_cfg: StudentVehicleMultiAgentGoalEnvCfg, runner_cfg: RslRlOnPolicyRunnerCfg
) -> dict[str, Any]:
    policy_type = str(_cfg_value(file_cfg, "policy", "type", "mlp")).strip().lower().replace("-", "_")
    policy_cfg: dict[str, Any] = {
        "type": policy_type,
        "class_name": str(runner_cfg.policy.class_name),
    }
    if policy_type == "late_fusion":
        policy_cfg.update(_build_late_fusion_policy_kwargs(env_cfg))
    return {
        "env": {
            "num_envs": int(env_cfg.scene.num_envs),
            "num_agents_per_env": int(env_cfg.num_agents_per_env),
            "observation_mode": str(env_cfg.observation_mode),
            "env_spacing": float(env_cfg.scene.env_spacing),
            "spawn_height_m": float(env_cfg.spawn_height_m),
            "decimation": int(env_cfg.decimation),
            "ground_mode": str(env_cfg.ground_mode),
            "apply_runtime_external_wrench": bool(env_cfg.apply_runtime_external_wrench),
            "use_scene_factory_roads": bool(env_cfg.use_scene_factory_roads),
            "start_radius_m": float(env_cfg.start_radius_m),
            "agent_spawn_circle_radius_m": float(env_cfg.agent_spawn_circle_radius_m),
            "agent_spawn_jitter_m": float(env_cfg.agent_spawn_jitter_m),
            "episode_length_s": float(env_cfg.episode_length_s),
            "reset_mode": str(env_cfg.reset_mode),
            "goal_radius_min_m": float(env_cfg.goal_radius_min_m),
            "goal_radius_max_m": float(env_cfg.goal_radius_max_m),
            "goal_reached_threshold_m": float(env_cfg.goal_reached_threshold_m),
            "max_distance_from_origin_m": float(env_cfg.max_distance_from_origin_m),
            "agent_neighbor_obs_scale_m": float(env_cfg.agent_neighbor_obs_scale_m),
            "agent_collision_warmup_steps": int(env_cfg.agent_collision_warmup_steps),
            "replicate_physics": bool(env_cfg.scene.replicate_physics),
            "clone_in_fabric": bool(env_cfg.scene.clone_in_fabric),
        },
        "scene_factory": {
            "config_path": str(env_cfg.scene_factory_config_path),
            "world_index": int(env_cfg.scene_factory_world_index),
            "world_selection_mode": str(env_cfg.scene_factory_world_selection_mode),
            "random_world_seed": int(env_cfg.scene_factory_random_world_seed),
        },
        "observation": {
            "weather_context_enable": bool(env_cfg.obs_weather_context_enable),
            "road_points_enable": bool(env_cfg.obs_road_points_enable),
            "road_points_k": int(env_cfg.obs_road_points_k),
            "road_points_radius_m": float(env_cfg.obs_road_points_radius_m),
            "road_points_type_norm": float(env_cfg.obs_road_points_type_norm),
            "road_points_mode": str(env_cfg.obs_road_points_mode),
            "road_points_include_dirs": bool(env_cfg.obs_road_points_include_dirs),
            "neighbor_enable": bool(env_cfg.obs_neighbor_enable),
            "neighbor_k": int(env_cfg.obs_neighbor_k),
            "neighbor_include_ttc": bool(env_cfg.obs_neighbor_include_ttc),
            "neighbor_include_index": bool(env_cfg.obs_neighbor_include_index),
            "neighbor_ttc_max_s": float(env_cfg.obs_neighbor_ttc_max_s),
            "timing_print_enable": bool(env_cfg.obs_timing_print_enable),
            "timing_print_every_n": int(env_cfg.obs_timing_print_every_n),
        },
        "timing": {
            "step_log_enable": bool(env_cfg.step_timing_log_enable),
            "step_print_enable": bool(env_cfg.step_timing_print_enable),
            "step_print_every_n": int(env_cfg.step_timing_print_every_n),
            "step_cuda_sync_enable": bool(env_cfg.step_timing_cuda_sync_enable),
        },
        "reward": {
            "mode": str(env_cfg.reward_mode),
            "lane_center_enable": bool(env_cfg.reward_lane_center_enable),
            "lane_center_types": [int(v) for v in env_cfg.reward_lane_center_types],
            "lane_center_per_step": float(env_cfg.reward_lane_center_per_step),
            "lane_forbidden_enable": bool(env_cfg.reward_lane_forbidden_enable),
            "lane_forbidden_types": [int(v) for v in env_cfg.reward_lane_forbidden_types],
            "lane_forbidden_penalty": float(env_cfg.reward_lane_forbidden_penalty),
            "collision_penalty": float(env_cfg.reward_collision_penalty),
            "crash_penalty": float(env_cfg.reward_crash_penalty),
            "goal_bonus": float(env_cfg.reward_goal_bonus),
            "choco_offroad_penalty": float(env_cfg.reward_choco_offroad_penalty),
            "choco_idle_penalty_enable": bool(env_cfg.reward_choco_idle_penalty_enable),
            "choco_idle_penalty_per_step": float(env_cfg.reward_choco_idle_penalty_per_step),
            "choco_idle_speed_threshold_mps": float(env_cfg.reward_choco_idle_speed_threshold_mps),
            "choco_speed_bonus_enable": bool(env_cfg.reward_choco_speed_bonus_enable),
            "choco_speed_bonus_per_step": float(env_cfg.reward_choco_speed_bonus_per_step),
            "choco_speed_bonus_max_mps": float(env_cfg.reward_choco_speed_bonus_max_mps),
            "choco_geom_lane_enable": bool(env_cfg.reward_choco_geom_lane_enable),
            "choco_geom_lane_per_step": float(env_cfg.reward_choco_geom_lane_per_step),
            "choco_geom_lane_tolerance_m": float(env_cfg.reward_choco_geom_lane_tolerance_m),
            "choco_geom_lane_heading_weight": float(env_cfg.reward_choco_geom_lane_heading_weight),
            "choco_geom_lane_min_alignment": float(env_cfg.reward_choco_geom_lane_min_alignment),
            "choco_geom_route_progress_weight": float(env_cfg.reward_choco_geom_route_progress_weight),
            "choco_geom_offroad_enable": bool(env_cfg.reward_choco_geom_offroad_enable),
            "choco_geom_offroad_lateral_threshold_m": float(env_cfg.reward_choco_geom_offroad_lateral_threshold_m),
            "choco_geom_offroad_distance_threshold_m": float(env_cfg.reward_choco_geom_offroad_distance_threshold_m),
            "choco_ttc_penalty_enable": bool(env_cfg.reward_choco_ttc_penalty_enable),
            "choco_ttc_penalty_alpha": float(env_cfg.reward_choco_ttc_penalty_alpha),
            "choco_ttc_penalty_max": float(env_cfg.reward_choco_ttc_penalty_max),
            "choco_ttc_penalty_min_ttc": float(env_cfg.reward_choco_ttc_penalty_min_ttc),
            "choco_road_edge_ttc_penalty_enable": bool(env_cfg.reward_choco_road_edge_ttc_penalty_enable),
            "choco_road_edge_ttc_penalty_alpha": float(env_cfg.reward_choco_road_edge_ttc_penalty_alpha),
            "choco_road_edge_ttc_penalty_max": float(env_cfg.reward_choco_road_edge_ttc_penalty_max),
            "choco_road_edge_ttc_penalty_min_ttc": float(env_cfg.reward_choco_road_edge_ttc_penalty_min_ttc),
            "choco_road_edge_ttc_hard_min_ttc": float(env_cfg.reward_choco_road_edge_ttc_hard_min_ttc),
            "choco_road_edge_ttc_radius_m": float(env_cfg.reward_choco_road_edge_ttc_radius_m),
        },
        "policy": policy_cfg,
        "test": {
            "mode": str(env_cfg.test_mode),
            "checkpoint_path": str(Path(args_cli.checkpoint_path).expanduser().resolve()) if str(args_cli.checkpoint_path).strip() else "",
            "max_steps": int(args_cli.eval_max_steps),
            "double_time_allowance": bool(args_cli.double_time_allowance),
            "invincible": bool(args_cli.invincible),
            "collision_force_threshold_n": float(env_cfg.agent_collision_force_threshold_n),
            "post_collision_steps": int(env_cfg.collision_test_post_collision_steps),
            "post_collision_throttle": float(env_cfg.collision_test_post_collision_throttle),
            "post_collision_steering": float(env_cfg.collision_test_post_collision_steering),
            "post_collision_brake": float(env_cfg.collision_test_post_collision_brake),
            "settle_steps": int(env_cfg.random_steer_test_settle_steps),
            "drive_steps": int(env_cfg.random_steer_test_drive_steps),
            "throttle": float(env_cfg.random_steer_test_throttle),
            "brake": float(env_cfg.random_steer_test_brake),
            "steering_min": float(env_cfg.random_steer_test_steering_min),
            "steering_max": float(env_cfg.random_steer_test_steering_max),
            "steering_hold_steps": int(env_cfg.random_steer_test_steering_hold_steps),
            "seed": int(env_cfg.random_steer_test_seed),
        },
        "assets": {
            "student_usd": str(env_cfg.student_usd_path),
            "tunable_config_json": str(env_cfg.tunable_config_json),
        },
        "runner": {
            "log_dir": str(Path(args_cli.log_dir).expanduser().resolve()),
            "seed": int(env_cfg.seed),
            "experiment_name": str(runner_cfg.experiment_name),
            "run_name": str(runner_cfg.run_name),
            "shared_policy_mode": str(args_cli.shared_policy_mode),
            "max_iterations": int(runner_cfg.max_iterations),
            "num_steps_per_env": int(runner_cfg.num_steps_per_env),
            "save_interval": int(runner_cfg.save_interval),
            "learning_rate": float(runner_cfg.algorithm.learning_rate),
            "num_learning_epochs": int(runner_cfg.algorithm.num_learning_epochs),
            "num_mini_batches": int(runner_cfg.algorithm.num_mini_batches),
            "entropy_coef": float(runner_cfg.algorithm.entropy_coef),
            "clip_param": float(runner_cfg.algorithm.clip_param),
            "desired_kl": float(runner_cfg.algorithm.desired_kl),
            "gamma": float(runner_cfg.algorithm.gamma),
            "gae_lambda": float(runner_cfg.algorithm.lam),
        },
        "video": {
            "enabled": bool(args_cli.video),
            "interval": int(args_cli.video_interval),
            "length": int(args_cli.video_length),
            "name_prefix": str(args_cli.video_name_prefix),
            "width": int(args_cli.video_width),
            "height": int(args_cli.video_height),
            "fps": int(args_cli.video_fps),
            "step_stride": int(args_cli.video_step_stride),
            "view_mode": str(args_cli.video_view_mode),
            "env_index": int(args_cli.video_env_index),
            "vehicle_proxy_markers": bool(args_cli.video_vehicle_proxy_markers),
            "vehicle_proxy_z_offset_m": float(args_cli.video_vehicle_proxy_z_offset_m),
        },
        "app": {
            "device": str(env_cfg.sim.device),
            "rl_device": str(runner_cfg.device),
            "headless": bool(getattr(args_cli, "headless", False)),
            "enable_cameras": bool(getattr(args_cli, "enable_cameras", False)),
            "silence_runtime_warnings": bool(getattr(args_cli, "silence_runtime_warnings", False)),
        },
    }


def _write_run_metadata(run_dir: Path, env_cfg: StudentVehicleMultiAgentGoalEnvCfg, runner_cfg: RslRlOnPolicyRunnerCfg):
    (run_dir / "params").mkdir(parents=True, exist_ok=True)
    resolved_cfg = _build_resolved_config(env_cfg, runner_cfg)
    policy_type = str(_cfg_value(file_cfg, "policy", "type", "mlp")).strip().lower().replace("-", "_")
    policy_payload: dict[str, Any] = {
        "type": policy_type,
        "class_name": str(runner_cfg.policy.class_name),
    }
    if policy_type == "late_fusion":
        policy_payload.update(_build_late_fusion_policy_kwargs(env_cfg))
    payload = {
        "config_path": str(Path(args_cli.config).expanduser().resolve()),
        "command": sys.orig_argv,
        "env_cfg": {
            "num_envs": env_cfg.scene.num_envs,
            "num_agents_per_env": env_cfg.num_agents_per_env,
            "sim_device": env_cfg.sim.device,
            "student_usd_path": env_cfg.student_usd_path,
            "tunable_config_json": env_cfg.tunable_config_json,
            "spawn_height_m": env_cfg.spawn_height_m,
            "decimation": env_cfg.decimation,
            "ground_mode": env_cfg.ground_mode,
            "apply_runtime_external_wrench": env_cfg.apply_runtime_external_wrench,
            "use_scene_factory_roads": env_cfg.use_scene_factory_roads,
            "scene_factory_config_path": env_cfg.scene_factory_config_path,
            "scene_factory_world_index": env_cfg.scene_factory_world_index,
            "scene_factory_world_selection_mode": env_cfg.scene_factory_world_selection_mode,
            "scene_factory_random_world_seed": env_cfg.scene_factory_random_world_seed,
            "test_mode": env_cfg.test_mode,
            "checkpoint_path": str(Path(args_cli.checkpoint_path).expanduser().resolve()) if str(args_cli.checkpoint_path).strip() else "",
            "eval_max_steps": int(args_cli.eval_max_steps),
            "double_time_allowance": bool(args_cli.double_time_allowance),
            "invincible": bool(args_cli.invincible),
            "collision_test_post_collision_steps": env_cfg.collision_test_post_collision_steps,
            "collision_test_post_collision_throttle": env_cfg.collision_test_post_collision_throttle,
            "collision_test_post_collision_steering": env_cfg.collision_test_post_collision_steering,
            "collision_test_post_collision_brake": env_cfg.collision_test_post_collision_brake,
            "random_steer_test_settle_steps": env_cfg.random_steer_test_settle_steps,
            "random_steer_test_drive_steps": env_cfg.random_steer_test_drive_steps,
            "random_steer_test_throttle": env_cfg.random_steer_test_throttle,
            "random_steer_test_brake": env_cfg.random_steer_test_brake,
            "random_steer_test_steering_min": env_cfg.random_steer_test_steering_min,
            "random_steer_test_steering_max": env_cfg.random_steer_test_steering_max,
            "random_steer_test_steering_hold_steps": env_cfg.random_steer_test_steering_hold_steps,
            "random_steer_test_seed": env_cfg.random_steer_test_seed,
            "start_radius_m": env_cfg.start_radius_m,
            "agent_spawn_circle_radius_m": env_cfg.agent_spawn_circle_radius_m,
            "agent_spawn_jitter_m": env_cfg.agent_spawn_jitter_m,
            "env_spacing": env_cfg.scene.env_spacing,
            "replicate_physics": env_cfg.scene.replicate_physics,
            "clone_in_fabric": env_cfg.scene.clone_in_fabric,
            "agent_collision_force_threshold_n": env_cfg.agent_collision_force_threshold_n,
            "agent_collision_warmup_steps": env_cfg.agent_collision_warmup_steps,
            "episode_length_s": env_cfg.episode_length_s,
            "reset_mode": env_cfg.reset_mode,
            "goal_radius_min_m": env_cfg.goal_radius_min_m,
            "goal_radius_max_m": env_cfg.goal_radius_max_m,
            "goal_reached_threshold_m": env_cfg.goal_reached_threshold_m,
            "max_distance_from_origin_m": env_cfg.max_distance_from_origin_m,
            "agent_neighbor_obs_scale_m": env_cfg.agent_neighbor_obs_scale_m,
            "observation_mode": env_cfg.observation_mode,
            "obs_weather_context_enable": env_cfg.obs_weather_context_enable,
            "obs_road_points_enable": env_cfg.obs_road_points_enable,
            "obs_road_points_k": env_cfg.obs_road_points_k,
            "obs_road_points_radius_m": env_cfg.obs_road_points_radius_m,
            "obs_road_points_type_norm": env_cfg.obs_road_points_type_norm,
            "obs_road_points_mode": env_cfg.obs_road_points_mode,
            "obs_road_points_include_dirs": env_cfg.obs_road_points_include_dirs,
            "obs_neighbor_enable": env_cfg.obs_neighbor_enable,
            "obs_neighbor_k": env_cfg.obs_neighbor_k,
            "obs_neighbor_include_ttc": env_cfg.obs_neighbor_include_ttc,
            "obs_neighbor_include_index": env_cfg.obs_neighbor_include_index,
            "obs_neighbor_ttc_max_s": env_cfg.obs_neighbor_ttc_max_s,
            "obs_timing_print_enable": env_cfg.obs_timing_print_enable,
            "obs_timing_print_every_n": env_cfg.obs_timing_print_every_n,
            "step_timing_log_enable": env_cfg.step_timing_log_enable,
            "step_timing_print_enable": env_cfg.step_timing_print_enable,
            "step_timing_print_every_n": env_cfg.step_timing_print_every_n,
            "step_timing_cuda_sync_enable": env_cfg.step_timing_cuda_sync_enable,
            "reward_lane_center_enable": env_cfg.reward_lane_center_enable,
            "reward_lane_center_types": list(env_cfg.reward_lane_center_types),
            "reward_lane_center_per_step": env_cfg.reward_lane_center_per_step,
            "reward_lane_forbidden_enable": env_cfg.reward_lane_forbidden_enable,
            "reward_lane_forbidden_types": list(env_cfg.reward_lane_forbidden_types),
            "reward_lane_forbidden_penalty": env_cfg.reward_lane_forbidden_penalty,
            "reward_collision_penalty": env_cfg.reward_collision_penalty,
            "reward_crash_penalty": env_cfg.reward_crash_penalty,
            "reward_mode": env_cfg.reward_mode,
            "reward_goal_bonus": env_cfg.reward_goal_bonus,
            "reward_choco_offroad_penalty": env_cfg.reward_choco_offroad_penalty,
            "reward_choco_idle_penalty_enable": env_cfg.reward_choco_idle_penalty_enable,
            "reward_choco_idle_penalty_per_step": env_cfg.reward_choco_idle_penalty_per_step,
            "reward_choco_idle_speed_threshold_mps": env_cfg.reward_choco_idle_speed_threshold_mps,
            "reward_choco_speed_bonus_enable": env_cfg.reward_choco_speed_bonus_enable,
            "reward_choco_speed_bonus_per_step": env_cfg.reward_choco_speed_bonus_per_step,
            "reward_choco_speed_bonus_max_mps": env_cfg.reward_choco_speed_bonus_max_mps,
            "reward_choco_geom_lane_enable": env_cfg.reward_choco_geom_lane_enable,
            "reward_choco_geom_lane_per_step": env_cfg.reward_choco_geom_lane_per_step,
            "reward_choco_geom_lane_tolerance_m": env_cfg.reward_choco_geom_lane_tolerance_m,
            "reward_choco_geom_lane_heading_weight": env_cfg.reward_choco_geom_lane_heading_weight,
            "reward_choco_geom_lane_min_alignment": env_cfg.reward_choco_geom_lane_min_alignment,
            "reward_choco_geom_offroad_enable": env_cfg.reward_choco_geom_offroad_enable,
            "reward_choco_geom_offroad_lateral_threshold_m": env_cfg.reward_choco_geom_offroad_lateral_threshold_m,
            "reward_choco_geom_offroad_distance_threshold_m": env_cfg.reward_choco_geom_offroad_distance_threshold_m,
            "reward_choco_ttc_penalty_enable": env_cfg.reward_choco_ttc_penalty_enable,
            "reward_choco_ttc_penalty_alpha": env_cfg.reward_choco_ttc_penalty_alpha,
            "reward_choco_ttc_penalty_max": env_cfg.reward_choco_ttc_penalty_max,
            "reward_choco_ttc_penalty_min_ttc": env_cfg.reward_choco_ttc_penalty_min_ttc,
            "reward_choco_road_edge_ttc_penalty_enable": env_cfg.reward_choco_road_edge_ttc_penalty_enable,
            "reward_choco_road_edge_ttc_penalty_alpha": env_cfg.reward_choco_road_edge_ttc_penalty_alpha,
            "reward_choco_road_edge_ttc_penalty_max": env_cfg.reward_choco_road_edge_ttc_penalty_max,
            "reward_choco_road_edge_ttc_penalty_min_ttc": env_cfg.reward_choco_road_edge_ttc_penalty_min_ttc,
            "reward_choco_road_edge_ttc_hard_min_ttc": env_cfg.reward_choco_road_edge_ttc_hard_min_ttc,
            "reward_choco_road_edge_ttc_radius_m": env_cfg.reward_choco_road_edge_ttc_radius_m,
        },
        "runner_cfg": {
            **runner_cfg.to_dict(),
            "shared_policy_mode": str(args_cli.shared_policy_mode),
        },
        "policy_cfg": policy_payload,
        "video_cfg": {
            "enabled": bool(args_cli.video),
            "interval": int(args_cli.video_interval),
            "length": int(args_cli.video_length),
            "name_prefix": str(args_cli.video_name_prefix),
            "width": int(args_cli.video_width),
            "height": int(args_cli.video_height),
            "fps": int(args_cli.video_fps),
            "step_stride": int(args_cli.video_step_stride),
            "view_mode": str(args_cli.video_view_mode),
            "env_index": int(args_cli.video_env_index),
            "vehicle_proxy_markers": bool(args_cli.video_vehicle_proxy_markers),
            "vehicle_proxy_z_offset_m": float(args_cli.video_vehicle_proxy_z_offset_m),
        },
        "app_cfg": {
            "silence_runtime_warnings": bool(getattr(args_cli, "silence_runtime_warnings", False)),
        },
    }
    (run_dir / "params" / "run.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    dump_yaml(str(run_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(run_dir / "params" / "agent.yaml"), runner_cfg)
    dump_yaml(str(run_dir / "params" / "resolved_config.yaml"), resolved_cfg)


def _maybe_save_stage_usd(save_stage_usd: str) -> None:
    if not str(save_stage_usd).strip():
        return
    import omni.usd

    save_path = Path(save_stage_usd).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    usd_context = omni.usd.get_context()
    stage = usd_context.get_stage()
    if stage is None:
        raise RuntimeError("Unable to access current USD stage for stage export.")
    print(f"[INFO][SceneFactory] Exporting initialized training stage to {save_path} ...", flush=True)
    start_time = time.time()
    stage.Export(str(save_path))
    print(
        f"[INFO][SceneFactory] Saved initialized training stage to {save_path} "
        f"in {time.time() - start_time:.2f}s",
        flush=True,
    )


def _aggregate_agent_log_dict(
    extras: dict,
    *,
    agent_ids: list[str],
    device: torch.device | str,
) -> dict:
    if not isinstance(extras, dict):
        return extras
    if "log" in extras or "episode" in extras:
        return extras

    aggregated_sums: dict[str, torch.Tensor] = {}
    aggregated_counts: dict[str, int] = {}
    for agent_id in agent_ids:
        agent_extra = extras.get(agent_id)
        if not isinstance(agent_extra, dict):
            continue
        agent_log = agent_extra.get("log") or agent_extra.get("episode")
        if not isinstance(agent_log, dict):
            continue
        for key, value in agent_log.items():
            if isinstance(value, torch.Tensor):
                tensor_value = value.detach().to(device=device, dtype=torch.float32).reshape(-1)
            else:
                try:
                    tensor_value = torch.tensor([float(value)], device=device, dtype=torch.float32)
                except (TypeError, ValueError):
                    continue
            aggregated_sums[key] = aggregated_sums.get(key, torch.zeros((), device=device, dtype=torch.float32)) + tensor_value.sum()
            aggregated_counts[key] = aggregated_counts.get(key, 0) + int(tensor_value.numel())

    if not aggregated_sums:
        return extras

    aggregated_log = {
        key: aggregated_sums[key] / max(1, aggregated_counts[key])
        for key in aggregated_sums.keys()
    }
    merged_extras = dict(extras)
    merged_extras["log"] = aggregated_log
    return merged_extras


def _patch_single_agent_marl_observation_bridge(env) -> None:
    """Patch Isaac Lab's MARL->single-agent adapter with _get_observations for RSL-RL.

    Isaac Lab's multi_agent_to_single_agent helper provides reset/step but does not implement
    _get_observations(), while RslRlVecEnvWrapper calls it during runner initialization.
    """

    if not hasattr(env, "env") or not isinstance(env.env, DirectMARLEnv):
        return

    env_cls = type(env)

    def _get_observations(self):
        if getattr(self, "_state_as_observation", False):
            return {"policy": self.env.state()}
        obs = self.env._get_observations()
        return {
            "policy": torch.cat(
                [obs[agent].reshape(self.num_envs, -1) for agent in self.env.possible_agents],
                dim=-1,
            )
        }

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, extras = self.env.reset(seed, options)
        if getattr(self, "_state_as_observation", False):
            obs = {"policy": self.env.state()}
        else:
            obs = {
                "policy": torch.cat(
                    [obs[agent].reshape(self.num_envs, -1) for agent in self.env.possible_agents],
                    dim=-1,
                )
            }
        return obs, _aggregate_agent_log_dict(
            extras,
            agent_ids=list(getattr(self.env, "possible_agents", [])),
            device=self.env.device,
        )

    def step(self, action: torch.Tensor):
        index = 0
        _actions = {}
        for agent in self.env.possible_agents:
            delta = gym.spaces.flatdim(self.env.action_spaces[agent])
            _actions[agent] = action[:, index : index + delta]
            index += delta

        obs, rewards, terminated, time_outs, extras = self.env.step(_actions)
        if getattr(self, "_state_as_observation", False):
            obs = {"policy": self.env.state()}
        else:
            obs = {
                "policy": torch.cat(
                    [obs[agent].reshape(self.num_envs, -1) for agent in self.env.possible_agents],
                    dim=-1,
                )
            }

        rewards = sum(rewards.values())
        terminated = math.prod(terminated.values()).to(dtype=torch.bool)
        time_outs = math.prod(time_outs.values()).to(dtype=torch.bool)
        return obs, rewards, terminated, time_outs, _aggregate_agent_log_dict(
            extras,
            agent_ids=list(getattr(self.env, "possible_agents", [])),
            device=self.env.device,
        )

    def __getattr__(self, key: str):
        return getattr(self.env, key)

    env._get_observations = types.MethodType(_get_observations, env)
    env.reset = types.MethodType(reset, env)
    env.step = types.MethodType(step, env)
    env_cls.__getattr__ = __getattr__
    env_cls.episode_length_buf = property(
        lambda self: self.env.episode_length_buf,
        lambda self, value: setattr(self.env, "episode_length_buf", value),
    )


class AgentSlotSharedPolicyVecEnv(VecEnv):
    """Expose one PPO slot per agent while keeping the underlying world env shared.

    This mirrors the old gpudrive_choco shared-policy setup more closely than the
    joint-world concatenation bridge. Each slot corresponds to a fixed (world, agent)
    pair, while the underlying Isaac Lab env still performs resets at the world level.
    """

    def __init__(self, env: StudentVehicleMultiAgentGoalEnv, clip_actions: float | None = None):
        self.env = env
        self.clip_actions = clip_actions
        self.cfg = env.cfg
        self.device = env.device
        self.num_worlds = int(env.num_envs)
        self.agent_ids = list(env.possible_agents)
        self.num_agents_per_world = len(self.agent_ids)
        self.num_envs = self.num_worlds * self.num_agents_per_world
        self.max_episode_length = env.max_episode_length
        self.num_actions = gym.spaces.flatdim(env.action_spaces[self.agent_ids[0]])
        self._slot_world_indices = torch.arange(self.num_worlds, device=self.device).repeat_interleave(
            self.num_agents_per_world
        )
        self._slot_episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._slot_dead_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.single_action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions,),
            dtype=float,
        )
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.single_observation_space = None
        self.observation_space = None
        obs_dict, _ = self.env.reset()
        flat_obs = self._flatten_obs_dict(obs_dict)
        obs_dim = int(flat_obs.shape[-1])
        self.single_observation_space = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(obs_dim,),
            dtype=float,
        )
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self._slot_dead_mask = self._flatten_agent_done_mask()
        print(
            "[INFO][SceneFactory] Using per-agent shared-policy wrapper: "
            f"worlds={self.num_worlds} agents_per_world={self.num_agents_per_world} slots={self.num_envs}",
            flush=True,
        )

    @property
    def unwrapped(self) -> StudentVehicleMultiAgentGoalEnv:
        return self.env

    @property
    def render_mode(self) -> str | None:
        return getattr(self.env, "render_mode", None)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf.repeat_interleave(self.num_agents_per_world)

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        world_lengths = value.view(self.num_worlds, self.num_agents_per_world)[:, 0]
        self.env.episode_length_buf = world_lengths.to(
            device=self.env.episode_length_buf.device,
            dtype=self.env.episode_length_buf.dtype,
        )

    def seed(self, seed: int = -1) -> int:
        return self.env.seed(seed)

    def _flatten_obs_dict(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack([obs_dict[agent_id].reshape(self.num_worlds, -1) for agent_id in self.agent_ids], dim=1)
        return stacked.reshape(self.num_envs, -1)

    def _flatten_reward_or_done_dict(self, payload: dict[str, torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack([payload[agent_id].reshape(self.num_worlds) for agent_id in self.agent_ids], dim=1)
        return stacked.reshape(self.num_envs)

    def _flatten_agent_done_mask(self) -> torch.Tensor:
        return self.env._agent_done_mask.transpose(0, 1).reshape(self.num_envs)

    def _reshape_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        action_view = actions.view(self.num_worlds, self.num_agents_per_world, -1)
        return {
            agent_id: action_view[:, agent_idx, :]
            for agent_idx, agent_id in enumerate(self.agent_ids)
        }

    def reset(self) -> tuple[TensorDict, dict]:
        obs_dict, extras = self.env.reset()
        self._slot_dead_mask.zero_()
        self._slot_episode_length_buf.zero_()
        flat_obs = self._flatten_obs_dict(obs_dict)
        return TensorDict({"policy": flat_obs}, batch_size=[self.num_envs]), _aggregate_agent_log_dict(
            extras,
            agent_ids=self.agent_ids,
            device=self.device,
        )

    def get_observations(self) -> TensorDict:
        obs_dict = self.env._get_observations()
        return TensorDict({"policy": self._flatten_obs_dict(obs_dict)}, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        step_start = perf_counter()
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        prev_dead_mask = self._flatten_agent_done_mask()
        prev_world_episode_length = self.env.episode_length_buf.clone()

        obs_dict, reward_dict, terminated_dict, truncated_dict, extras = self.env.step(self._reshape_actions(actions))

        flat_obs = self._flatten_obs_dict(obs_dict)
        flat_rewards = self._flatten_reward_or_done_dict(reward_dict)
        flat_terminated = self._flatten_reward_or_done_dict(terminated_dict).to(dtype=torch.bool)
        flat_truncated = self._flatten_reward_or_done_dict(truncated_dict).to(dtype=torch.bool)

        world_reset_mask = self.env.episode_length_buf < (prev_world_episode_length + 1)
        flat_world_reset_mask = world_reset_mask.repeat_interleave(self.num_agents_per_world)
        newly_done = (flat_terminated | flat_truncated) & (~prev_dead_mask)
        dones = newly_done | flat_world_reset_mask

        flat_rewards = torch.where(prev_dead_mask, torch.zeros_like(flat_rewards), flat_rewards)
        self._slot_episode_length_buf += 1
        self._slot_episode_length_buf[dones] = 0
        self._slot_dead_mask = self._flatten_agent_done_mask()

        extras_agg_start = perf_counter()
        merged_extras = _aggregate_agent_log_dict(
            extras,
            agent_ids=self.agent_ids,
            device=self.device,
        )
        wrapper_total_ms = (perf_counter() - step_start) * 1000.0
        wrapper_aggregate_ms = (perf_counter() - extras_agg_start) * 1000.0
        controlled_agent_count = float(
            self.env._spawned_agent_mask.sum().item()
            if hasattr(self.env, "_spawned_agent_mask")
            else self.num_envs
        )
        env_steps_per_s = 1000.0 / max(wrapper_total_ms, 1.0e-6)
        controlled_agent_steps_per_s = controlled_agent_count * env_steps_per_s
        if bool(getattr(self.env.cfg, "step_timing_log_enable", False)):
            log = merged_extras.get("log") if isinstance(merged_extras, dict) else None
            if not isinstance(log, dict):
                log = {}
                if not isinstance(merged_extras, dict):
                    merged_extras = {}
                merged_extras["log"] = log
            log["Perf/wrapper_step_total_ms"] = float(wrapper_total_ms)
            log["Perf/wrapper_aggregate_extras_ms"] = float(wrapper_aggregate_ms)
            log["Perf/controlled_agent_count"] = float(controlled_agent_count)
            log["Perf/env_steps_per_s"] = float(env_steps_per_s)
            log["Perf/controlled_agent_steps_per_s"] = float(controlled_agent_steps_per_s)
            log["Perf/CASPS"] = float(controlled_agent_steps_per_s)
        if not bool(getattr(self.env.cfg, "is_finite_horizon", True)):
            merged_extras["time_outs"] = flat_truncated

        return (
            TensorDict({"policy": flat_obs}, batch_size=[self.num_envs]),
            flat_rewards,
            dones.to(dtype=torch.long),
            merged_extras,
        )

    def close(self):
        return self.env.close()


class SensorVideoRecorderWrapper(gym.Wrapper):
    """Record periodic training clips from a fixed Isaac Lab camera sensor."""

    def __init__(self, env, capture_env: StudentVehicleMultiAgentGoalEnv, run_dir: Path):
        super().__init__(env)
        self._capture_env = capture_env
        self._video_dir = run_dir / "videos" / "train"
        self._video_dir.mkdir(parents=True, exist_ok=True)
        self._interval = max(1, int(args_cli.video_interval))
        self._length = max(1, int(args_cli.video_length))
        self._fps = max(1, int(args_cli.video_fps))
        self._step_stride = max(1, int(args_cli.video_step_stride))
        self._name_prefix = str(args_cli.video_name_prefix)
        self._global_step = 0
        self._clip_index = 0
        self._recording = False
        self._recording_step_index = 0
        self._frames: list = []

    def _start_clip(self) -> None:
        if self._recording:
            return
        self._recording = True
        self._recording_step_index = 0
        self._frames = []

    def _capture_frame(self, *, force: bool = False) -> None:
        if not self._recording:
            return
        should_capture = force or (self._recording_step_index % self._step_stride == 0)
        self._recording_step_index += 1
        if not should_capture:
            return
        frame = self._capture_env.capture_fixed_camera_frame()
        if frame is not None:
            self._frames.append(frame)
        if len(self._frames) >= self._length:
            self._finish_clip()

    def _finish_clip(self) -> None:
        if not self._recording:
            return
        self._recording = False
        if len(self._frames) == 0:
            self._frames = []
            return
        import imageio.v2 as imageio

        output_path = self._video_dir / f"{self._name_prefix}_{self._clip_index:04d}.mp4"
        with imageio.get_writer(str(output_path), fps=self._fps) as writer:
            for frame in self._frames:
                writer.append_data(frame)
        self._clip_index += 1
        self._frames = []

    def reset(self, **kwargs):
        output = self.env.reset(**kwargs)
        if self._global_step % self._interval == 0:
            self._start_clip()
        self._capture_frame(force=True)
        return output

    def step(self, action):
        output = self.env.step(action)
        self._global_step += 1
        if (not self._recording) and self._global_step % self._interval == 0:
            self._start_clip()
        self._capture_frame()
        return output

    def close(self):
        self._finish_clip()
        return super().close()


def _maybe_wrap_video(env, capture_env: StudentVehicleMultiAgentGoalEnv, run_dir: Path):
    if not bool(args_cli.video):
        return env
    return SensorVideoRecorderWrapper(env, capture_env=capture_env, run_dir=run_dir)


def _collision_test_action_dict(
    env: StudentVehicleMultiAgentGoalEnv,
    throttle: float,
    steering: float,
    brake: float,
) -> dict[str, torch.Tensor]:
    action = torch.tensor(
        [float(throttle), float(steering), float(brake)],
        dtype=torch.float32,
        device=env.device,
    ).unsqueeze(0).repeat(env.num_envs, 1)
    return {agent_id: action.clone() for agent_id in env.cfg.possible_agents}


def _action_dict_from_components(
    env: StudentVehicleMultiAgentGoalEnv,
    throttle_by_agent: torch.Tensor,
    steering_by_agent: torch.Tensor,
    brake_by_agent: torch.Tensor,
) -> dict[str, torch.Tensor]:
    action_dict: dict[str, torch.Tensor] = {}
    for agent_idx, agent_id in enumerate(env.cfg.possible_agents):
        action_dict[agent_id] = torch.stack(
            [
                throttle_by_agent[agent_idx],
                steering_by_agent[agent_idx],
                brake_by_agent[agent_idx],
            ],
            dim=-1,
        ).to(device=env.device, dtype=torch.float32)
    return action_dict


def _run_collision_test(env: StudentVehicleMultiAgentGoalEnv, run_dir: Path) -> None:
    import imageio.v2 as imageio

    test_mode_name = str(env.cfg.test_mode).strip().lower() or "collision_test"
    metrics_path = run_dir / f"{test_mode_name}_metrics.jsonl"
    summary_path = run_dir / f"{test_mode_name}_summary.json"
    video_path = run_dir / "videos" / f"{test_mode_name}.mp4"
    video_step_stride = max(1, int(args_cli.video_step_stride))
    video_writer = None
    if bool(args_cli.video):
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(str(video_path), fps=max(1, int(args_cli.video_fps)))
    video_capture_step_index = 0
    frames_written = 0

    def _write_frame(*, force: bool = False) -> None:
        nonlocal video_capture_step_index, frames_written
        if video_writer is None:
            return
        should_capture = force or (video_capture_step_index % video_step_stride == 0)
        video_capture_step_index += 1
        if not should_capture:
            return
        frame = env.capture_fixed_camera_frame()
        if frame is not None:
            video_writer.append_data(frame)
            frames_written += 1

    print(f"[INFO][SceneFactory] Running deterministic {test_mode_name} rollout.", flush=True)
    obs, extras = env.reset()
    _write_frame(force=True)
    collision_step: int | None = None
    max_collision_force_n = 0.0
    raw_max_collision_force_n = 0.0
    collision_detection_armed_step = int(env.cfg.collision_test_settle_steps)
    total_steps = int(
        env.cfg.collision_test_settle_steps
        + env.cfg.collision_test_drive_steps
        + env.cfg.collision_test_post_collision_steps
    )
    post_collision_steps_remaining = int(env.cfg.collision_test_post_collision_steps)

    with metrics_path.open("w", encoding="utf-8") as handle:
        for step in range(total_steps):
            if step < int(env.cfg.collision_test_settle_steps):
                action_dict = _collision_test_action_dict(env, throttle=0.0, steering=0.0, brake=0.0)
            elif collision_step is not None:
                action_dict = _collision_test_action_dict(
                    env,
                    throttle=float(env.cfg.collision_test_post_collision_throttle),
                    steering=float(env.cfg.collision_test_post_collision_steering),
                    brake=float(env.cfg.collision_test_post_collision_brake),
                )
            else:
                action_dict = _collision_test_action_dict(
                    env,
                    throttle=float(env.cfg.collision_test_throttle),
                    steering=float(env.cfg.collision_test_steering),
                    brake=float(env.cfg.collision_test_brake),
                )

            obs, rewards, terminated, time_outs, extras = env.step(action_dict)
            _write_frame()

            collision_force_by_agent = env.collision_force_by_agent_n()
            collision_world_force = env.collision_world_force_n()
            lane_touch_types_by_agent = env.lane_touch_types_by_agent()
            raw_collision_detected = bool(
                torch.any(collision_world_force >= float(env.cfg.agent_collision_force_threshold_n))
            )
            raw_max_collision_force_n = max(raw_max_collision_force_n, float(collision_world_force.max().item()))
            collision_detected = raw_collision_detected and step >= collision_detection_armed_step
            if collision_detected:
                max_collision_force_n = max(max_collision_force_n, float(collision_world_force.max().item()))
            if collision_detected and collision_step is None:
                collision_step = step
                print(
                    f"[INFO][SceneFactory] {test_mode_name} detected contact at "
                    f"step={step} force_n={float(collision_world_force.max().item()):.2f}",
                    flush=True,
                )
            elif collision_step is not None:
                post_collision_steps_remaining -= 1

            step_record: dict[str, Any] = {
                "step": int(step),
                "collision_detection_armed": bool(step >= collision_detection_armed_step),
                "raw_collision_detected": raw_collision_detected,
                "collision_detected": collision_detected,
                "collision_step": collision_step,
                "post_collision_steps_remaining": int(max(post_collision_steps_remaining, 0)),
                "collision_world_force_n": [float(x) for x in collision_world_force.detach().cpu().tolist()],
                "terminated": {agent_id: bool(terminated[agent_id][0].item()) for agent_id in env.cfg.possible_agents},
                "time_outs": {agent_id: bool(time_outs[agent_id][0].item()) for agent_id in env.cfg.possible_agents},
                "rewards": {agent_id: float(rewards[agent_id][0].item()) for agent_id in env.cfg.possible_agents},
                "agents": {},
            }
            for agent_idx, agent_id in enumerate(env.cfg.possible_agents):
                vehicle = env._vehicles[agent_idx]
                root_pos_w = vehicle.data.root_pos_w[0]
                root_lin_vel_w = vehicle.data.root_lin_vel_w[0]
                step_record["agents"][agent_id] = {
                    "root_pos_w": [float(x) for x in root_pos_w.detach().cpu().tolist()],
                    "root_lin_vel_w": [float(x) for x in root_lin_vel_w.detach().cpu().tolist()],
                    "planar_speed_mps": float(torch.linalg.norm(root_lin_vel_w[:2]).item()),
                    "goal_distance_m": float(env._current_goal_distance[agent_idx, 0].item()),
                    "collision_force_n": float(collision_force_by_agent[agent_id][0].item()),
                    "lane_touch_types": list(lane_touch_types_by_agent.get(agent_id, [[]])[0]),
            }
            handle.write(json.dumps(step_record) + "\n")

            if collision_step is not None and post_collision_steps_remaining <= 0:
                break

    if video_writer is not None:
        video_writer.close()

    summary = {
        "test_mode": test_mode_name,
        "collision_detection_armed_step": int(collision_detection_armed_step),
        "collision_detected": collision_step is not None,
        "collision_step": collision_step,
        "max_collision_force_n": max_collision_force_n,
        "raw_max_collision_force_n": raw_max_collision_force_n,
        "frames_written": int(frames_written),
        "video_step_stride": int(video_step_stride),
        "video_fps": int(args_cli.video_fps) if bool(args_cli.video) else 0,
        "video_duration_s": float(frames_written / max(1, int(args_cli.video_fps))) if bool(args_cli.video) else 0.0,
        "metrics_path": str(metrics_path),
        "video_path": str(video_path) if bool(args_cli.video) else "",
        "config_path": str(Path(args_cli.config).expanduser().resolve()),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"[INFO][SceneFactory] {test_mode_name} finished. "
        f"collision_detected={summary['collision_detected']} max_force_n={max_collision_force_n:.2f}",
        flush=True,
    )


def _run_bicycle_sinwave_demo(env: StudentVehicleMultiAgentGoalEnv, run_dir: Path) -> None:
    """Run a scripted sin-wave throttle + steering demo using the bicycle dynamics model.

    The action schedule is fully deterministic and hand-authored:
      - Phase 1 (settle):   0 throttle, 0 steer  — vehicles settle at spawn
      - Phase 2 (straight): constant throttle, 0 steer
      - Phase 3 (sinwave):  constant throttle, sin-wave steering with per-agent phase offset
                            so neighbouring vehicles curve in opposite directions, making the
                            physics-gap visible (no suspension roll, no lateral weight transfer).
    """
    import math as _math
    import imageio.v2 as imageio

    test_mode_name = "bicycle_sinwave_demo"
    video_path = run_dir / "videos" / f"{test_mode_name}.mp4"
    video_step_stride = max(1, int(args_cli.video_step_stride))
    video_writer = None
    if bool(args_cli.video):
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(str(video_path), fps=max(1, int(args_cli.video_fps)))
    video_capture_step_index = 0

    def _write_frame(*, force: bool = False) -> None:
        nonlocal video_capture_step_index
        if video_writer is None:
            return
        should_capture = force or (video_capture_step_index % video_step_stride == 0)
        video_capture_step_index += 1
        if not should_capture:
            return
        frame = env.capture_fixed_camera_frame()
        if frame is not None:
            video_writer.append_data(frame)

    # ── Scripted schedule parameters ─────────────────────────────────────
    SETTLE_STEPS   = 20    # idle at spawn, let physics/bicycle settle
    STRAIGHT_STEPS = 30    # accelerate straight before weaving
    SINWAVE_STEPS  = 300   # main sinusoidal weave phase

    THROTTLE       = 0.65  # constant forward drive [-1, 1]
    STEER_AMP      = 0.60  # sin-wave steering amplitude
    STEER_PERIOD   = 60    # steps per full steering cycle

    num_agents = env._num_agents
    num_envs   = env.num_envs
    device     = env.device

    print(
        f"[INFO][BicycleSinwaveDemo] settle={SETTLE_STEPS} straight={STRAIGHT_STEPS} "
        f"sinwave={SINWAVE_STEPS} throttle={THROTTLE} steer_amp={STEER_AMP} period={STEER_PERIOD}",
        flush=True,
    )

    obs, extras = env.reset()
    _write_frame(force=True)

    total_steps = SETTLE_STEPS + STRAIGHT_STEPS + SINWAVE_STEPS
    for step in range(total_steps):
        if step < SETTLE_STEPS:
            # --- Phase 1: idle ---
            throttle = torch.zeros((num_agents, num_envs), dtype=torch.float32, device=device)
            steer    = torch.zeros((num_agents, num_envs), dtype=torch.float32, device=device)
            brake    = torch.zeros((num_agents, num_envs), dtype=torch.float32, device=device)

        elif step < SETTLE_STEPS + STRAIGHT_STEPS:
            # --- Phase 2: straight acceleration ---
            throttle = torch.full((num_agents, num_envs), THROTTLE,  dtype=torch.float32, device=device)
            steer    = torch.zeros((num_agents, num_envs), dtype=torch.float32, device=device)
            brake    = torch.zeros((num_agents, num_envs), dtype=torch.float32, device=device)

        else:
            # --- Phase 3: sin-wave weave with per-agent phase offset ---
            t = step - SETTLE_STEPS - STRAIGHT_STEPS
            throttle = torch.full((num_agents, num_envs), THROTTLE, dtype=torch.float32, device=device)
            brake    = torch.zeros((num_agents, num_envs), dtype=torch.float32, device=device)
            steer    = torch.zeros((num_agents, num_envs), dtype=torch.float32, device=device)
            for agent_idx in range(num_agents):
                # each agent gets a different phase so they don't all turn together
                phase_offset = _math.pi * agent_idx / max(1, num_agents)
                steer_val = STEER_AMP * _math.sin(2.0 * _math.pi * t / STEER_PERIOD + phase_offset)
                steer[agent_idx, :] = float(steer_val)

        action_dict = _action_dict_from_components(env, throttle, steer, brake)
        obs, rewards, terminated, time_outs, extras = env.step(action_dict)
        _write_frame()

        if step % 50 == 0:
            print(f"[BicycleSinwaveDemo] step {step}/{total_steps}", flush=True)

    if video_writer is not None:
        video_writer.close()
        print(f"[INFO][BicycleSinwaveDemo] Video saved → {video_path}", flush=True)


def _run_scene_factory_multiworld_random_steer_test(env: StudentVehicleMultiAgentGoalEnv, run_dir: Path) -> None:
    import imageio.v2 as imageio

    test_mode_name = str(env.cfg.test_mode).strip().lower() or "scene_factory_multiworld_random_steer_test"
    metrics_path = run_dir / f"{test_mode_name}_metrics.jsonl"
    summary_path = run_dir / f"{test_mode_name}_summary.json"
    video_path = run_dir / "videos" / f"{test_mode_name}.mp4"
    video_step_stride = max(1, int(args_cli.video_step_stride))
    video_writer = None
    if bool(args_cli.video):
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(str(video_path), fps=max(1, int(args_cli.video_fps)))
    video_capture_step_index = 0
    frames_written = 0

    def _write_frame(*, force: bool = False) -> None:
        nonlocal video_capture_step_index, frames_written
        if video_writer is None:
            return
        should_capture = force or (video_capture_step_index % video_step_stride == 0)
        video_capture_step_index += 1
        if not should_capture:
            return
        frame = env.capture_fixed_camera_frame()
        if frame is not None:
            video_writer.append_data(frame)
            frames_written += 1

    print(f"[INFO][SceneFactory] Running deterministic {test_mode_name} rollout.", flush=True)
    obs, extras = env.reset()
    _write_frame(force=True)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(env.cfg.random_steer_test_seed))
    total_steps = int(env.cfg.random_steer_test_settle_steps + env.cfg.random_steer_test_drive_steps)
    hold_steps = max(1, int(env.cfg.random_steer_test_steering_hold_steps))
    steering_min = float(env.cfg.random_steer_test_steering_min)
    steering_max = float(env.cfg.random_steer_test_steering_max)
    max_collision_force_n = 0.0
    collision_step_count = 0
    worlds_with_collision: set[int] = set()
    lane_types_touched_global: set[int] = set()
    current_steering = torch.zeros((env._num_agents, env.num_envs), dtype=torch.float32, device=env.device)

    with metrics_path.open("w", encoding="utf-8") as handle:
        for step in range(total_steps):
            if step < int(env.cfg.random_steer_test_settle_steps):
                throttle_by_agent = torch.zeros((env._num_agents, env.num_envs), dtype=torch.float32, device=env.device)
                steering_by_agent = torch.zeros((env._num_agents, env.num_envs), dtype=torch.float32, device=env.device)
                brake_by_agent = torch.zeros((env._num_agents, env.num_envs), dtype=torch.float32, device=env.device)
            else:
                if (step - int(env.cfg.random_steer_test_settle_steps)) % hold_steps == 0:
                    current_steering = (
                        steering_min
                        + (steering_max - steering_min)
                        * torch.rand((env._num_agents, env.num_envs), generator=generator, dtype=torch.float32)
                    ).to(env.device)
                throttle_by_agent = torch.full(
                    (env._num_agents, env.num_envs),
                    float(env.cfg.random_steer_test_throttle),
                    dtype=torch.float32,
                    device=env.device,
                )
                steering_by_agent = current_steering
                brake_by_agent = torch.full(
                    (env._num_agents, env.num_envs),
                    float(env.cfg.random_steer_test_brake),
                    dtype=torch.float32,
                    device=env.device,
                )

            action_dict = _action_dict_from_components(env, throttle_by_agent, steering_by_agent, brake_by_agent)
            obs, rewards, terminated, time_outs, extras = env.step(action_dict)
            _write_frame()

            collision_force_by_agent = env.collision_force_by_agent_n()
            collision_world_force = env.collision_world_force_n()
            lane_touch_types_by_agent = env.lane_touch_types_by_agent()
            collision_world_mask = collision_world_force >= float(env.cfg.agent_collision_force_threshold_n)
            if bool(torch.any(collision_world_mask).item()):
                collision_step_count += 1
                hit_env_ids = torch.nonzero(collision_world_mask, as_tuple=False).view(-1).tolist()
                for env_id in hit_env_ids:
                    worlds_with_collision.add(int(env_id))
            max_collision_force_n = max(max_collision_force_n, float(collision_world_force.max().item()))

            step_record: dict[str, Any] = {
                "step": int(step),
                "collision_world_force_n": [float(x) for x in collision_world_force.detach().cpu().tolist()],
                "collision_world_mask": [bool(x) for x in collision_world_mask.detach().cpu().tolist()],
                "envs": [],
            }
            for env_idx in range(env.num_envs):
                env_record: dict[str, Any] = {
                    "env_index": int(env_idx),
                    "collision_world_force_n": float(collision_world_force[env_idx].item()),
                    "collision_world": bool(collision_world_mask[env_idx].item()),
                    "agents": {},
                }
                for agent_idx, agent_id in enumerate(env.cfg.possible_agents):
                    vehicle = env._vehicles[agent_idx]
                    root_pos_w = vehicle.data.root_pos_w[env_idx]
                    root_lin_vel_w = vehicle.data.root_lin_vel_w[env_idx]
                    lane_types = list(lane_touch_types_by_agent.get(agent_id, [[]])[env_idx])
                    lane_types_touched_global.update(int(t) for t in lane_types)
                    env_record["agents"][agent_id] = {
                        "root_pos_w": [float(x) for x in root_pos_w.detach().cpu().tolist()],
                        "root_lin_vel_w": [float(x) for x in root_lin_vel_w.detach().cpu().tolist()],
                        "planar_speed_mps": float(torch.linalg.norm(root_lin_vel_w[:2]).item()),
                        "goal_distance_m": float(env._current_goal_distance[agent_idx, env_idx].item()),
                        "collision_force_n": float(collision_force_by_agent[agent_id][env_idx].item()),
                        "lane_touch_types": lane_types,
                        "steering_cmd": float(steering_by_agent[agent_idx, env_idx].item()),
                    }
                step_record["envs"].append(env_record)
            handle.write(json.dumps(step_record) + "\n")

    if video_writer is not None:
        video_writer.close()

    summary = {
        "test_mode": test_mode_name,
        "num_envs": int(env.num_envs),
        "num_agents_per_env": int(env._num_agents),
        "total_steps": int(total_steps),
        "collision_step_count": int(collision_step_count),
        "worlds_with_collision": sorted(int(x) for x in worlds_with_collision),
        "world_collision_count": int(len(worlds_with_collision)),
        "max_collision_force_n": float(max_collision_force_n),
        "lane_types_touched_global": sorted(int(x) for x in lane_types_touched_global),
        "frames_written": int(frames_written),
        "video_step_stride": int(video_step_stride),
        "video_fps": int(args_cli.video_fps) if bool(args_cli.video) else 0,
        "video_duration_s": float(frames_written / max(1, int(args_cli.video_fps))) if bool(args_cli.video) else 0.0,
        "metrics_path": str(metrics_path),
        "video_path": str(video_path) if bool(args_cli.video) else "",
        "config_path": str(Path(args_cli.config).expanduser().resolve()),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"[INFO][SceneFactory] {test_mode_name} finished. "
        f"world_collision_count={summary['world_collision_count']} "
        f"max_force_n={summary['max_collision_force_n']:.2f} "
        f"lane_types={summary['lane_types_touched_global']}",
        flush=True,
    )


def _run_scene_factory_policy_eval(
    base_env: StudentVehicleMultiAgentGoalEnv,
    env,
    runner: OnPolicyRunner,
    run_dir: Path,
) -> None:
    import imageio.v2 as imageio

    checkpoint_path_raw = str(args_cli.checkpoint_path).strip()
    _has_scripted = (
        args_cli.fixed_action is not None
        or args_cli.action_schedule is not None
        or bool(getattr(base_env.cfg, "fixed_action", ""))
        or bool(getattr(base_env.cfg, "action_schedule", ""))
    )
    if not checkpoint_path_raw and not _has_scripted:
        raise ValueError(
            "scene_factory_policy_eval requires --checkpoint_path or test.checkpoint_path in the config."
        )
    checkpoint_path = Path(checkpoint_path_raw).expanduser().resolve() if checkpoint_path_raw else None
    if checkpoint_path is not None and not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    test_mode_name = str(base_env.cfg.test_mode).strip().lower() or "scene_factory_policy_eval"
    steps_path = run_dir / f"{test_mode_name}_steps.jsonl"
    worlds_path = run_dir / f"{test_mode_name}_worlds.jsonl"
    summary_path = run_dir / f"{test_mode_name}_summary.json"
    video_path = run_dir / "videos" / f"{test_mode_name}.mp4"
    video_step_stride = max(1, int(args_cli.video_step_stride))
    video_writer = None
    video_writers_per_env: list = []  # for per_env mode
    is_per_env_video = str(args_cli.video_view_mode).strip().lower() == "per_env"
    if bool(args_cli.video):
        video_path.parent.mkdir(parents=True, exist_ok=True)
        if is_per_env_video:
            for ei in range(int(base_env.num_envs)):
                p = video_path.parent / f"{test_mode_name}_env{ei}.mp4"
                video_writers_per_env.append(
                    imageio.get_writer(str(p), fps=max(1, int(args_cli.video_fps)))
                )
        else:
            video_writer = imageio.get_writer(str(video_path), fps=max(1, int(args_cli.video_fps)))
    video_capture_step_index = 0

    def _write_frame(*, force: bool = False) -> None:
        nonlocal video_capture_step_index, frames_written
        if video_writer is None and not video_writers_per_env:
            return
        should_capture = force or (video_capture_step_index % video_step_stride == 0)
        video_capture_step_index += 1
        if not should_capture:
            return
        if video_writers_per_env:
            per_env_frames = base_env.capture_per_env_frames()
            for ei, (writer, frame) in enumerate(zip(video_writers_per_env, per_env_frames)):
                if frame is not None:
                    writer.append_data(frame)
                    frames_written += 1
        elif video_writer is not None:
            frame = base_env.capture_fixed_camera_frame()
            if frame is not None:
                video_writer.append_data(frame)
                frames_written += 1

    print(
        f"[INFO][SceneFactory] Running deterministic {test_mode_name} rollout from checkpoint {checkpoint_path}.",
        flush=True,
    )
    if bool(getattr(base_env.cfg, "invincible", False)):
        print(
            "[INFO][SceneFactory] scene_factory_policy_eval invincible=true: "
            "collision/crash/forbidden-lane events are logged but do not terminate or clear vehicles.",
            flush=True,
        )
    if bool(getattr(base_env.cfg, "random_od", False)):
        print(
            f"[INFO][SceneFactory] scene_factory_policy_eval random_od=true: "
            f"OD pairs will be randomly resampled on lane centerlines at each episode reset "
            f"(travel={base_env.cfg.random_od_min_travel_m:.0f}-{base_env.cfg.random_od_max_travel_m:.0f}m).",
            flush=True,
        )

    # ── Parse action schedule (CLI overrides config) ──
    _action_schedule = None
    _action_schedule_raw = None
    if args_cli.action_schedule is not None:
        _action_schedule_raw = args_cli.action_schedule  # list of "step:t,s,b" strings
    elif getattr(base_env.cfg, "action_schedule", ""):
        _action_schedule_raw = str(base_env.cfg.action_schedule).strip().split()

    if _action_schedule_raw:
        _action_schedule = []
        for entry in _action_schedule_raw:
            step_str, vals_str = entry.split(":")
            vals = [float(v) for v in vals_str.split(",")]
            assert len(vals) == 3, f"Expected 3 values (throttle,steer,brake) in '{entry}'"
            _action_schedule.append((int(step_str), vals))
        _action_schedule.sort(key=lambda x: x[0])
        print(
            f"[INFO][SceneFactory] action_schedule with {len(_action_schedule)} entries: "
            + ", ".join(f"step{s}→[{a[0]:.2f},{a[1]:.2f},{a[2]:.2f}]" for s, a in _action_schedule),
            flush=True,
        )

    # ── Parse fixed_action (CLI overrides config) ──
    _fixed_action = None
    if args_cli.fixed_action is not None:
        _fixed_action = args_cli.fixed_action
    elif getattr(base_env.cfg, "fixed_action", ""):
        _fixed_action = [float(v) for v in str(base_env.cfg.fixed_action).strip().split(",")]

    if _action_schedule is not None or _fixed_action is not None:
        _label = "action_schedule" if _action_schedule is not None else f"fixed_action={_fixed_action}"
        print(
            f"[INFO][SceneFactory] {_label} specified; "
            f"skipping checkpoint load (policy output will be overridden).",
            flush=True,
        )
        inference_policy = runner.get_inference_policy(device=str(runner.device))
    else:
        runner.load(str(checkpoint_path), load_optimizer=False, map_location=str(runner.device))
        inference_policy = runner.get_inference_policy(device=str(runner.device))

    base_env.consume_last_reset_world_episode_summaries()
    obs, extras = env.reset()
    base_env.consume_last_reset_world_episode_summaries()
    base_env.episode_length_buf.zero_()
    if hasattr(base_env, "_steps_since_reset_buf"):
        base_env._steps_since_reset_buf.zero_()
    if hasattr(env, "_slot_episode_length_buf"):
        env._slot_episode_length_buf.zero_()
    if hasattr(env, "_slot_dead_mask") and hasattr(env, "_flatten_agent_done_mask"):
        env._slot_dead_mask = env._flatten_agent_done_mask()

    completed_worlds: dict[int, dict[str, Any]] = {}
    max_steps = int(args_cli.eval_max_steps)
    if max_steps <= 0:
        max_steps = int(base_env.max_episode_length) + 1
    frames_written = 0
    _write_frame(force=True)

    with steps_path.open("w", encoding="utf-8") as steps_handle, worlds_path.open("w", encoding="utf-8") as worlds_handle:
        with torch.inference_mode():
            for step in range(max_steps):
                actions = inference_policy(obs)
                if _action_schedule is not None:
                    # find the last entry whose step <= current step
                    _cur_act = _action_schedule[0][1]
                    for _t, _a in _action_schedule:
                        if step >= _t:
                            _cur_act = _a
                        else:
                            break
                    actions = torch.tensor(
                        [_cur_act], device=actions.device, dtype=actions.dtype
                    ).expand_as(actions)
                elif _fixed_action is not None:
                    actions = torch.tensor(
                        [_fixed_action], device=actions.device, dtype=actions.dtype
                    ).expand_as(actions)
                step_output = env.step(actions)
                if len(step_output) == 4:
                    obs, rewards, dones, extras = step_output
                else:
                    raise RuntimeError(
                        f"Unexpected env.step output length for {test_mode_name}: {len(step_output)}"
                    )

                _write_frame()

                # ── Friction ruler: log per-env velocity every 30 steps ──
                if bool(getattr(base_env.cfg, "friction_ruler_mode", False)) and step % 30 == 0:
                    for vi, vehicle in enumerate(base_env._vehicles):
                        vel = vehicle.data.root_lin_vel_w  # (num_envs, 3)
                        speed = vel.norm(dim=-1)  # (num_envs,)
                        pos_y = vehicle.data.root_pos_w[:, 1] - base_env.scene.env_origins[:, 1]
                        parts = []
                        for ei in range(min(int(base_env.num_envs), 8)):
                            parts.append(
                                f"env{ei}: speed={speed[ei].item():.3f}m/s  y={pos_y[ei].item():.2f}m"
                            )
                        print(f"  [RULER] step={step}  {' | '.join(parts)}", flush=True)

                new_world_summaries = base_env.consume_last_reset_world_episode_summaries()
                new_completed_envs: list[int] = []
                for world_summary in new_world_summaries:
                    env_index = int(world_summary.get("env_index", -1))
                    if env_index < 0 or env_index in completed_worlds:
                        continue
                    completed_worlds[env_index] = dict(world_summary)
                    new_completed_envs.append(env_index)
                    worlds_handle.write(json.dumps(world_summary) + "\n")

                if isinstance(rewards, torch.Tensor):
                    mean_reward = float(rewards.detach().float().mean().item())
                else:
                    mean_reward = float(torch.as_tensor(rewards, dtype=torch.float32).mean().item())
                if isinstance(dones, torch.Tensor):
                    done_count = int(dones.detach().to(dtype=torch.int64).sum().item())
                else:
                    done_count = int(torch.as_tensor(dones, dtype=torch.int64).sum().item())

                step_record = {
                    "step": int(step),
                    "mean_slot_reward": float(mean_reward),
                    "done_slot_count": int(done_count),
                    "completed_world_count": int(len(completed_worlds)),
                    "new_completed_env_indices": [int(v) for v in sorted(new_completed_envs)],
                }
                steps_handle.write(json.dumps(step_record) + "\n")

                if len(completed_worlds) >= int(base_env.num_envs):
                    break

    if video_writer is not None:
        video_writer.close()
    for w in video_writers_per_env:
        w.close()

    completed_items = [completed_worlds[idx] for idx in sorted(completed_worlds.keys())]
    total_spawned = float(sum(float(item.get("spawned_count", 0.0)) for item in completed_items))
    total_success = float(sum(float(item.get("success_count", 0.0)) for item in completed_items))
    total_collision = float(sum(float(item.get("collision_count", 0.0)) for item in completed_items))
    total_lane_forbidden = float(sum(float(item.get("lane_forbidden_count", 0.0)) for item in completed_items))
    total_crash = float(sum(float(item.get("crash_count", 0.0)) for item in completed_items))
    total_crash_too_low = float(sum(float(item.get("crash_too_low_count", 0.0)) for item in completed_items))
    total_crash_too_far = float(sum(float(item.get("crash_too_far_count", 0.0)) for item in completed_items))
    total_crash_bad_tilt = float(sum(float(item.get("crash_bad_tilt_count", 0.0)) for item in completed_items))
    total_active_not_done = float(sum(float(item.get("active_not_done_count", 0.0)) for item in completed_items))
    total_spawned_denom = max(1.0, total_spawned)
    mean_final_distance_to_goal = (
        float(sum(float(item.get("mean_final_distance_to_goal", 0.0)) for item in completed_items) / max(1, len(completed_items)))
        if completed_items
        else 0.0
    )
    # TTC / DRAC safety metrics — only average over worlds where they were computed (value >= 0)
    _ttc_worlds = [item for item in completed_items if float(item.get("mean_min_ttc_s", -1.0)) >= 0.0]
    mean_min_ttc_s = float(sum(float(item["mean_min_ttc_s"]) for item in _ttc_worlds) / max(1, len(_ttc_worlds))) if _ttc_worlds else -1.0
    near_miss_rate = float(sum(float(item["near_miss_rate"]) for item in _ttc_worlds) / max(1, len(_ttc_worlds))) if _ttc_worlds else -1.0
    mean_max_drac = float(sum(float(item["mean_max_drac"]) for item in _ttc_worlds) / max(1, len(_ttc_worlds))) if _ttc_worlds else -1.0
    high_drac_rate = float(sum(float(item["high_drac_rate"]) for item in _ttc_worlds) / max(1, len(_ttc_worlds))) if _ttc_worlds else -1.0
    per_world_summary_sorted = sorted(
        [
            {
                "env_index": int(item.get("env_index", -1)),
                "world_index": int(item.get("world_index", -1)),
                "scene_json_name": str(item.get("scene_json_name", "")),
                "spawned_count": float(item.get("spawned_count", 0.0)),
                "success_count": float(item.get("success_count", 0.0)),
                "collision_count": float(item.get("collision_count", 0.0)),
                "lane_forbidden_count": float(item.get("lane_forbidden_count", 0.0)),
                "crash_count": float(item.get("crash_count", 0.0)),
                "active_not_done_count": float(item.get("active_not_done_count", 0.0)),
                "success_rate": float(item.get("success_rate", 0.0)),
                "collision_rate": float(item.get("collision_rate", 0.0)),
                "lane_forbidden_rate": float(item.get("lane_forbidden_rate", 0.0)),
                "crash_rate": float(item.get("crash_rate", 0.0)),
                "mean_final_distance_to_goal": float(item.get("mean_final_distance_to_goal", 0.0)),
                "episode_length_steps": int(item.get("episode_length_steps", 0)),
                "mean_min_ttc_s": float(item.get("mean_min_ttc_s", -1.0)),
                "near_miss_rate": float(item.get("near_miss_rate", -1.0)),
                "mean_max_drac": float(item.get("mean_max_drac", -1.0)),
                "high_drac_rate": float(item.get("high_drac_rate", -1.0)),
            }
            for item in completed_items
        ],
        key=lambda item: (
            float(item["success_rate"]),
            float(item["success_count"]),
            -float(item["mean_final_distance_to_goal"]),
            -float(item["lane_forbidden_rate"]),
            -float(item["collision_rate"]),
            -float(item["crash_rate"]),
            int(item["env_index"]),
        ),
    )
    summary = {
        "test_mode": test_mode_name,
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(Path(args_cli.config).expanduser().resolve()),
        "invincible": bool(getattr(base_env.cfg, "invincible", False)),
        "random_od": bool(getattr(base_env.cfg, "random_od", False)),
        "random_od_min_travel_m": float(getattr(base_env.cfg, "random_od_min_travel_m", 20.0)),
        "random_od_max_travel_m": float(getattr(base_env.cfg, "random_od_max_travel_m", 60.0)),
        "double_time_allowance": bool(args_cli.double_time_allowance),
        "max_steps": int(max_steps),
        "completed_world_count": int(len(completed_items)),
        "expected_world_count": int(base_env.num_envs),
        "all_worlds_completed": bool(len(completed_items) >= int(base_env.num_envs)),
        "total_spawned_count": float(total_spawned),
        "total_success_count": float(total_success),
        "total_collision_count": float(total_collision),
        "total_lane_forbidden_count": float(total_lane_forbidden),
        "total_crash_count": float(total_crash),
        "total_crash_too_low_count": float(total_crash_too_low),
        "total_crash_too_far_count": float(total_crash_too_far),
        "total_crash_bad_tilt_count": float(total_crash_bad_tilt),
        "total_active_not_done_count": float(total_active_not_done),
        "success_rate": float(total_success / total_spawned_denom),
        "collision_rate": float(total_collision / total_spawned_denom),
        "lane_forbidden_rate": float(total_lane_forbidden / total_spawned_denom),
        "crash_rate": float(total_crash / total_spawned_denom),
        "crash_too_low_rate": float(total_crash_too_low / total_spawned_denom),
        "crash_too_far_rate": float(total_crash_too_far / total_spawned_denom),
        "crash_bad_tilt_rate": float(total_crash_bad_tilt / total_spawned_denom),
        "mean_final_distance_to_goal": float(mean_final_distance_to_goal),
        "mean_min_ttc_s": float(mean_min_ttc_s),
        "near_miss_rate": float(near_miss_rate),
        "mean_max_drac": float(mean_max_drac),
        "high_drac_rate": float(high_drac_rate),
        "steps_path": str(steps_path),
        "worlds_path": str(worlds_path),
        "video_path": str(video_path) if bool(args_cli.video) else "",
        "frames_written": int(frames_written),
        "video_step_stride": int(video_step_stride),
        "video_fps": int(args_cli.video_fps) if bool(args_cli.video) else 0,
        "video_duration_s": float(frames_written / max(1, int(args_cli.video_fps))) if bool(args_cli.video) else 0.0,
        "per_world_summary_sorted_by_least_success": per_world_summary_sorted,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _ttc_str = f" mean_min_ttc={summary['mean_min_ttc_s']:.2f}s near_miss_rate={summary['near_miss_rate']:.3f} mean_max_drac={summary['mean_max_drac']:.3f}m/s² high_drac_rate={summary['high_drac_rate']:.3f}" if summary['mean_min_ttc_s'] >= 0.0 else ""
    print(
        f"[INFO][SceneFactory] {test_mode_name} finished. "
        f"completed_worlds={summary['completed_world_count']}/{summary['expected_world_count']} "
        f"success_rate={summary['success_rate']:.3f} "
        f"collision_rate={summary['collision_rate']:.3f} "
        f"lane_forbidden_rate={summary['lane_forbidden_rate']:.3f} "
        f"crash_rate={summary['crash_rate']:.3f}"
        f"{_ttc_str}",
        flush=True,
    )
    if per_world_summary_sorted:
        print("[INFO][SceneFactory] Per-world summary (least successful first):", flush=True)
        for item in per_world_summary_sorted:
            print(
                "[INFO][SceneFactory] "
                f"env={item['env_index']:02d} "
                f"scene={item['scene_json_name']} "
                f"spawned={item['spawned_count']:.0f} "
                f"success={item['success_count']:.0f} ({item['success_rate']:.2f}) "
                f"collision={item['collision_count']:.0f} ({item['collision_rate']:.2f}) "
                f"lane_forbidden={item['lane_forbidden_count']:.0f} ({item['lane_forbidden_rate']:.2f}) "
                f"crash={item['crash_count']:.0f} ({item['crash_rate']:.2f}) "
                f"active={item['active_not_done_count']:.0f} "
                f"final_dist={item['mean_final_distance_to_goal']:.2f}m "
                f"steps={item['episode_length_steps']}",
                flush=True,
            )


def main():
    env_cfg = _build_env_cfg()
    runner_cfg = _build_runner_cfg(env_cfg.sim.device)
    log_root = Path(args_cli.log_dir).expanduser().resolve()
    run_dir = _make_run_dir(log_root, runner_cfg)
    print(f"[INFO] Logging RSL-RL run in: {run_dir}")

    start_time = time.time()
    base_env = StudentVehicleMultiAgentGoalEnv(env_cfg, render_mode=None)
    _write_run_metadata(run_dir, env_cfg, runner_cfg)
    _maybe_save_stage_usd(args_cli.save_stage_usd)
    if bool(args_cli.exit_after_stage_save):
        base_env.close()
        print("[INFO][SceneFactory] Exiting after stage save as requested.")
        return
    if str(args_cli.test_mode).strip().lower() in {"collision_test", "scene_factory_collision_test"}:
        try:
            _run_collision_test(base_env, run_dir)
        finally:
            base_env.close()
            print(f"[INFO] Collision test finished in {time.time() - start_time:.2f}s")
        return
    if str(args_cli.test_mode).strip().lower() == "scene_factory_multiworld_random_steer_test":
        try:
            _run_scene_factory_multiworld_random_steer_test(base_env, run_dir)
        finally:
            base_env.close()
            print(f"[INFO] Random steer test finished in {time.time() - start_time:.2f}s")
        return
    if str(args_cli.test_mode).strip().lower() == "bicycle_sinwave_demo":
        try:
            _run_bicycle_sinwave_demo(base_env, run_dir)
        finally:
            base_env.close()
            print(f"[INFO] Bicycle sinwave demo finished in {time.time() - start_time:.2f}s")
        return

    if str(args_cli.shared_policy_mode).strip().lower() == "agent_slots":
        env = AgentSlotSharedPolicyVecEnv(base_env, clip_actions=runner_cfg.clip_actions)
    else:
        env = base_env
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
            _patch_single_agent_marl_observation_bridge(env)
        env = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)

    eval_test_mode = str(args_cli.test_mode).strip().lower() in {"scene_factory_policy_eval", "friction_ruler"}
    if not eval_test_mode:
        env = _maybe_wrap_video(env, capture_env=base_env, run_dir=run_dir)

    train_cfg = runner_cfg.to_dict()
    # Strip any keys in train_cfg["algorithm"] that PPO.__init__ doesn't accept.
    # Newer IsaacLab versions add fields (e.g. "optimizer") to RslRlPpoAlgorithmCfg
    # that older rsl_rl PPO does not recognise, causing a TypeError at runner init.
    if "algorithm" in train_cfg:
        import inspect as _inspect
        from rsl_rl.algorithms import PPO as _PPO
        _ppo_params = set(_inspect.signature(_PPO.__init__).parameters.keys()) - {"self", "policy"}
        _unknown = {k: v for k, v in train_cfg["algorithm"].items() if k not in _ppo_params and k != "class_name"}
        if _unknown:
            print(f"[INFO][SceneFactory] Dropping unknown PPO alg_cfg keys (rsl_rl version mismatch): {list(_unknown.keys())}", flush=True)
            for k in _unknown:
                train_cfg["algorithm"].pop(k)
    policy_type = str(_cfg_value(file_cfg, "policy", "type", "mlp")).strip().lower().replace("-", "_")
    if policy_type == "late_fusion":
        _register_scene_factory_custom_policy_classes()
        train_cfg["policy"]["class_name"] = "SceneFactoryLateFusionActorCritic"
        train_cfg["policy"].update(_build_late_fusion_policy_kwargs(env_cfg))
        print(
            "[INFO][SceneFactory] Using late-fusion actor-critic policy "
            f"with road_k={train_cfg['policy']['road_point_k']} vehicle_k={train_cfg['policy']['vehicle_k']}.",
            flush=True,
        )

    runner = OnPolicyRunner(env, train_cfg, log_dir=str(run_dir), device=str(runner_cfg.device))
    runner.git_status_repos = [__file__]

    resume_path_raw = str(getattr(args_cli, "resume_from", "")).strip()
    if resume_path_raw and not eval_test_mode:
        resume_path = Path(resume_path_raw).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"--resume_from checkpoint not found: {resume_path}")
        print(f"[INFO] Resuming training from checkpoint: {resume_path}", flush=True)
        runner.load(str(resume_path), load_optimizer=True)
        print(f"[INFO] Resumed at iteration {runner.current_learning_iteration}", flush=True)

    try:
        if eval_test_mode:
            _run_scene_factory_policy_eval(base_env, env, runner, run_dir)
        else:
            runner.learn(num_learning_iterations=int(runner_cfg.max_iterations), init_at_random_ep_len=True)
    finally:
        env.close()
        if eval_test_mode:
            print(f"[INFO] Policy eval finished in {time.time() - start_time:.2f}s")
        else:
            print(f"[INFO] Training finished in {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
    simulation_app.close()
