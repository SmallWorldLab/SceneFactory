from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
from datetime import datetime

from src.isaaclab_bootstrap import ensure_isaaclab_source_paths

ensure_isaaclab_source_paths()

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy for the student vehicle goal task.")
parser.add_argument("--run_dir", type=str, required=True, help="Training run directory containing model and params.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="model.zip",
    help="Checkpoint filename inside run_dir. Default uses the final model.zip.",
)
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel envs to visualize.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--num_steps", type=int, default=5000, help="Number of evaluation environment steps.")
parser.add_argument("--env_spacing", type=float, default=10.0, help="Spacing between visualized environments.")
parser.add_argument("--spawn_height_m", type=float, default=-1.0, help="Override spawn height. Negative keeps run cfg.")
parser.add_argument(
    "--ground_mode",
    choices=("plane", "cuboid"),
    default="",
    help="Optional override for the evaluation ground implementation. Empty keeps run cfg.",
)
parser.add_argument("--goal_radius_min_m", type=float, default=-1.0, help="Override min goal radius. Negative keeps run cfg.")
parser.add_argument("--goal_radius_max_m", type=float, default=-1.0, help="Override max goal radius. Negative keeps run cfg.")
parser.add_argument(
    "--deterministic",
    action="store_true",
    default=False,
    help="Use deterministic policy actions for evaluation.",
)
parser.add_argument(
    "--disable_debug_vis",
    action="store_true",
    default=False,
    help="Disable goal markers during evaluation.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from src.student_vehicle_goal_env import (
    StudentVehicleGoalEnv,
    StudentVehicleGoalEnvCfg,
    build_student_vehicle_articulation_cfg,
)


def _resolve_seed(seed: int) -> int:
    if int(seed) >= 0:
        return int(seed)
    return random.randint(0, 10_000)


def _load_run_metadata(run_dir: Path) -> dict:
    metadata_path = run_dir / "params" / "run.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing run metadata: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _build_env_cfg(run_dir: Path, metadata: dict) -> StudentVehicleGoalEnvCfg:
    env_meta = metadata.get("env_cfg", {})
    cfg = StudentVehicleGoalEnvCfg()
    cfg.seed = _resolve_seed(args_cli.seed)
    cfg.scene.num_envs = int(args_cli.num_envs)
    cfg.scene.env_spacing = float(args_cli.env_spacing)
    cfg.scene.clone_in_fabric = bool(getattr(args_cli, "headless", False))
    cfg.debug_vis = not bool(args_cli.disable_debug_vis)

    if args_cli.device is not None:
        cfg.sim.device = str(args_cli.device)
    else:
        cfg.sim.device = str(env_meta.get("sim_device", "cuda:0" if torch.cuda.is_available() else "cpu"))

    usd_path = str(env_meta.get("student_usd", ""))
    if not usd_path:
        raise ValueError("Run metadata does not contain env_cfg.student_usd")

    spawn_height_m = (
        float(args_cli.spawn_height_m) if float(args_cli.spawn_height_m) >= 0.0 else float(env_meta.get("spawn_height_m", 1.6))
    )
    cfg.spawn_height_m = spawn_height_m
    cfg.ground_mode = str(args_cli.ground_mode or env_meta.get("ground_mode", cfg.ground_mode))
    cfg.vehicle = build_student_vehicle_articulation_cfg(usd_path, spawn_height_m=spawn_height_m)

    cfg.tunable_config_json = str(env_meta.get("tunable_config_json", ""))
    cfg.episode_length_s = float(env_meta.get("episode_length_s", cfg.episode_length_s))
    cfg.goal_radius_min_m = (
        float(args_cli.goal_radius_min_m)
        if float(args_cli.goal_radius_min_m) >= 0.0
        else float(env_meta.get("goal_radius_min_m", cfg.goal_radius_min_m))
    )
    cfg.goal_radius_max_m = (
        float(args_cli.goal_radius_max_m)
        if float(args_cli.goal_radius_max_m) >= 0.0
        else float(env_meta.get("goal_radius_max_m", cfg.goal_radius_max_m))
    )
    return cfg


def _write_eval_metadata(run_dir: Path, env_cfg: StudentVehicleGoalEnvCfg, model_path: Path, vecnormalize_path: Path | None):
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    payload = {
        "command": sys.orig_argv,
        "model_path": str(model_path),
        "vecnormalize_path": str(vecnormalize_path) if vecnormalize_path else None,
        "num_envs": int(env_cfg.scene.num_envs),
        "sim_device": str(env_cfg.sim.device),
        "env_spacing": float(env_cfg.scene.env_spacing),
        "spawn_height_m": float(env_cfg.spawn_height_m),
        "ground_mode": str(env_cfg.ground_mode),
        "debug_vis": bool(env_cfg.debug_vis),
        "clone_in_fabric": bool(env_cfg.scene.clone_in_fabric),
        "num_steps": int(args_cli.num_steps),
        "deterministic": bool(args_cli.deterministic),
    }
    output_path = eval_dir / f"eval_{timestamp}.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def main():
    run_dir = Path(args_cli.run_dir).expanduser().resolve()
    metadata = _load_run_metadata(run_dir)
    model_path = run_dir / str(args_cli.checkpoint)
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")

    vecnormalize_path = run_dir / "model_vecnormalize.pkl"
    has_vecnormalize = bool(metadata.get("cli", {}).get("normalize_obs", False)) and vecnormalize_path.is_file()

    env_cfg = _build_env_cfg(run_dir, metadata)
    print(f"[INFO] Evaluating PPO policy from: {model_path}")
    env = StudentVehicleGoalEnv(env_cfg, render_mode=None)
    vec_env = Sb3VecEnvWrapper(env, fast_variant=False)

    if has_vecnormalize:
        vec_env = VecNormalize.load(str(vecnormalize_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    _write_eval_metadata(run_dir, env_cfg, model_path, vecnormalize_path if has_vecnormalize else None)

    agent = PPO.load(str(model_path), env=vec_env, device="auto")

    obs = vec_env.reset()
    episode_returns = torch.zeros(int(args_cli.num_envs), dtype=torch.float32)
    episode_lengths = torch.zeros(int(args_cli.num_envs), dtype=torch.int32)
    completed_returns: list[float] = []
    completed_lengths: list[int] = []

    start_time = time.time()
    try:
        for step_idx in range(int(args_cli.num_steps)):
            actions, _ = agent.predict(obs, deterministic=bool(args_cli.deterministic))
            obs, rewards, dones, infos = vec_env.step(actions)

            rewards_t = torch.as_tensor(rewards, dtype=torch.float32)
            dones_t = torch.as_tensor(dones, dtype=torch.bool)
            episode_returns += rewards_t
            episode_lengths += 1

            if torch.any(dones_t):
                done_ids = torch.nonzero(dones_t, as_tuple=False).squeeze(-1).tolist()
                for env_id in done_ids:
                    completed_returns.append(float(episode_returns[env_id].item()))
                    completed_lengths.append(int(episode_lengths[env_id].item()))
                    episode_returns[env_id] = 0.0
                    episode_lengths[env_id] = 0

            if (step_idx + 1) % 250 == 0:
                mean_return = sum(completed_returns) / len(completed_returns) if completed_returns else float("nan")
                mean_length = sum(completed_lengths) / len(completed_lengths) if completed_lengths else float("nan")
                print(
                    f"[INFO] Step {step_idx + 1}/{int(args_cli.num_steps)} | completed episodes={len(completed_returns)}"
                    f" | mean return={mean_return:.3f} | mean length={mean_length:.1f}"
                )
    finally:
        mean_return = sum(completed_returns) / len(completed_returns) if completed_returns else float("nan")
        mean_length = sum(completed_lengths) / len(completed_lengths) if completed_lengths else float("nan")
        print(
            f"[INFO] Evaluation finished in {time.time() - start_time:.2f}s"
            f" | completed episodes={len(completed_returns)} | mean return={mean_return:.3f}"
            f" | mean length={mean_length:.1f}"
        )
        vec_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
