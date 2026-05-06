from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import random

from src.isaaclab_bootstrap import ensure_isaaclab_source_paths

ensure_isaaclab_source_paths()

os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp_cache")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Run the multi-agent student-vehicle goal environment with held random actions."
)
parser.add_argument("--num_envs", type=int, default=2, help="Number of parallel world instances.")
parser.add_argument("--num_agents_per_env", type=int, default=14, help="Number of vehicles inside each world.")
parser.add_argument("--num_steps", type=int, default=1500, help="Number of environment steps to simulate.")
parser.add_argument("--action_hold_steps", type=int, default=12, help="Hold each sampled random action for N env steps.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--student_usd", type=str, default="", help="Path to the student vehicle USD.")
parser.add_argument(
    "--tunable_config_json",
    type=str,
    default="",
    help="Path to the tuned student config JSON. Empty uses the environment default.",
)
parser.add_argument("--spawn_height_m", type=float, default=1.6, help="Vehicle spawn height above each env origin.")
parser.add_argument("--ground_mode", choices=("plane", "cuboid"), default="plane")
parser.add_argument("--env_spacing", type=float, default=240.0, help="Spacing between vectorized environments.")
parser.add_argument("--start_radius_m", type=float, default=0.0, help="Shared per-world spawn offset radius.")
parser.add_argument("--agent_spawn_circle_radius_m", type=float, default=20.0, help="Within-world spawn ring radius.")
parser.add_argument("--agent_spawn_jitter_m", type=float, default=0.25, help="Per-agent spawn jitter.")
parser.add_argument("--episode_length_s", type=float, default=60.0, help="Episode length in seconds.")
parser.add_argument("--goal_radius_min_m", type=float, default=50.0, help="Minimum goal radius from env origin.")
parser.add_argument("--goal_radius_max_m", type=float, default=90.0, help="Maximum goal radius from env origin.")
parser.add_argument("--max_distance_from_origin_m", type=float, default=100.0, help="Logical world radius.")
parser.add_argument("--agent_neighbor_obs_scale_m", type=float, default=100.0, help="Neighbor observation scale.")
parser.add_argument("--replicate_physics", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--clone_in_fabric", choices=("auto", "true", "false"), default="auto")
parser.add_argument("--report_every", type=int, default=60, help="Print summary every N env steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

from src.student_vehicle_goal_env import DEFAULT_STUDENT_VEHICLE_USD
from src.student_vehicle_multiagent_goal_env import (
    StudentVehicleMultiAgentGoalEnv,
    StudentVehicleMultiAgentGoalEnvCfg,
    configure_multi_agent_spaces,
)


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
    cfg.debug_vis = not bool(args_cli.headless)
    if args_cli.device is not None:
        cfg.sim.device = str(args_cli.device)
    else:
        cfg.sim.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_gpu_device = not str(cfg.sim.device).lower().startswith("cpu")
    if args_cli.clone_in_fabric == "auto":
        cfg.scene.clone_in_fabric = (
            bool(getattr(args_cli, "headless", False)) and cfg.scene.replicate_physics and use_gpu_device
        )
    else:
        cfg.scene.clone_in_fabric = (
            args_cli.clone_in_fabric == "true" and cfg.scene.replicate_physics and use_gpu_device
        )
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
    cfg.ground_mode = str(args_cli.ground_mode)
    cfg.start_radius_m = float(args_cli.start_radius_m)
    cfg.agent_spawn_circle_radius_m = float(args_cli.agent_spawn_circle_radius_m)
    cfg.agent_spawn_jitter_m = float(args_cli.agent_spawn_jitter_m)
    cfg.episode_length_s = float(args_cli.episode_length_s)
    cfg.goal_radius_min_m = float(args_cli.goal_radius_min_m)
    cfg.goal_radius_max_m = float(args_cli.goal_radius_max_m)
    cfg.max_distance_from_origin_m = float(args_cli.max_distance_from_origin_m)
    cfg.agent_neighbor_obs_scale_m = float(args_cli.agent_neighbor_obs_scale_m)
    cfg.student_usd_path = str(Path(args_cli.student_usd or DEFAULT_STUDENT_VEHICLE_USD).expanduser().resolve())
    if str(args_cli.tunable_config_json):
        cfg.tunable_config_json = str(Path(args_cli.tunable_config_json).expanduser().resolve())
    configure_multi_agent_spaces(cfg, int(args_cli.num_agents_per_env))
    return cfg


def _sample_random_actions(env: StudentVehicleMultiAgentGoalEnv) -> dict[str, torch.Tensor]:
    actions: dict[str, torch.Tensor] = {}
    for agent_id in env.possible_agents:
        throttle = torch.rand(env.num_envs, 1, device=env.device)
        steering = 2.0 * torch.rand(env.num_envs, 1, device=env.device) - 1.0
        brake = torch.rand(env.num_envs, 1, device=env.device)
        use_brake = (torch.rand(env.num_envs, 1, device=env.device) < 0.25).float()
        throttle = throttle * (1.0 - use_brake)
        brake = brake * use_brake
        actions[agent_id] = torch.cat([throttle, steering, brake], dim=-1)
    return actions


def _report(env: StudentVehicleMultiAgentGoalEnv, step_idx: int) -> None:
    speeds = []
    distances = []
    for agent_idx in range(len(env.possible_agents)):
        speeds.append(torch.linalg.norm(env._vehicles[agent_idx].data.root_lin_vel_w[:, :2], dim=1))
        distances.append(env._current_goal_distance[agent_idx])
    mean_speed = float(torch.stack(speeds, dim=0).mean().item())
    mean_goal_distance = float(torch.stack(distances, dim=0).mean().item())
    print(
        f"[STEP {step_idx:05d}] mean_speed={mean_speed:.3f} m/s | "
        f"mean_goal_distance={mean_goal_distance:.3f} m"
    )


def main():
    env_cfg = _build_env_cfg()
    env = StudentVehicleMultiAgentGoalEnv(env_cfg, render_mode=None)
    actions = _sample_random_actions(env)
    env.reset()
    try:
        for step_idx in range(int(args_cli.num_steps)):
            if step_idx % max(1, int(args_cli.action_hold_steps)) == 0:
                actions = _sample_random_actions(env)
            env.step(actions)
            if (step_idx + 1) % max(1, int(args_cli.report_every)) == 0:
                _report(env, step_idx + 1)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
