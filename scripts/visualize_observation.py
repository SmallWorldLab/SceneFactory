"""
Observation visualizer for SceneFactory.

Thin wrapper around the training entry point that boots the env in
``obs_visualization`` test mode, steps it for a few warmup steps, extracts a
single agent's observation vector, and saves a 2-panel diagnostic figure to
``artifacts/obs_debug/<timestamp>_obs_viz.png``.

The figure shows:
  Left panel  — top-down map: road center-points, car position & heading, goal
                star, neighbor vehicle markers, all in ego-centered coordinates (m)
  Right panel — text display: ego scalars (speed, heading error, goal distance),
                weather token (h_w, road surface one-hot), road/neighbor counts

Usage
-----
From the repo root with PYTHONPATH=.:

    PYTHONPATH=. python scripts/visualize_observation.py \\
        --config configs/scene_factory/waymo_physx_256_train.yaml \\
        [--env_idx 0] \\
        [--agent_idx 0] \\
        [--num_warmup_steps 10] \\
        [--num_envs 4] \\
        [--headless]

All flags after the known ones are passed through to the training entry point.

Examples
--------
Visualize env 0, agent 0 with 4 worlds (small enough to be fast):
    PYTHONPATH=. python scripts/visualize_observation.py \\
        --config configs/scene_factory/demo_weather_physx_train.yaml \\
        --num_envs 4 --headless

Visualize env 2, agent 1 with extra warmup:
    PYTHONPATH=. python scripts/visualize_observation.py \\
        --config configs/scene_factory/waymo_physx_256_train.yaml \\
        --env_idx 2 --agent_idx 1 --num_warmup_steps 30 --num_envs 4 --headless
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the visualize_observation wrapper."""
    p = argparse.ArgumentParser(
        description="Visualize a SceneFactory observation vector as a 2-panel figure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
        add_help=True,
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/scene_factory/demo_weather_physx_train.yaml",
        help="Path to the training config YAML (e.g. configs/scene_factory/waymo_physx_256_train.yaml).",
    )
    p.add_argument(
        "--env_idx",
        type=int,
        default=0,
        help="Which parallel environment instance to visualize (default: 0).",
    )
    p.add_argument(
        "--agent_idx",
        type=int,
        default=0,
        help="Which vehicle slot within the chosen environment (default: 0).",
    )
    p.add_argument(
        "--num_warmup_steps",
        type=int,
        default=10,
        help="Warmup steps before capturing the observation (default: 10).",
    )
    p.add_argument(
        "--warmup_action",
        type=float,
        nargs=3,
        default=None,
        metavar=("THROTTLE", "STEER", "BRAKE"),
        help="Optional fixed [throttle, steer, brake] action for warmup; default is zero action.",
    )
    p.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help=(
            "Override num_envs from the config.  Useful to run a small subset of worlds "
            "for speed (e.g. --num_envs 4).  Must be > env_idx."
        ),
    )
    return p


def main() -> None:
    """Parse args and delegate to the training entry point in obs_visualization mode."""
    parser = _build_arg_parser()

    # Split known args from pass-through (e.g. --headless, --device, --num_agents_per_env …)
    known, passthrough = parser.parse_known_args()

    config_path = str(Path(known.config).expanduser().resolve())
    if not Path(config_path).is_file():
        sys.exit(f"[visualize_observation] Config not found: {config_path}")

    # Enforce env_idx constraint when num_envs is overridden
    if known.num_envs is not None and known.env_idx >= known.num_envs:
        sys.exit(
            f"[visualize_observation] --env_idx {known.env_idx} must be < "
            f"--num_envs {known.num_envs}"
        )

    train_script = str(
        Path(__file__).resolve().parent.parent / "src" / "train_student_vehicle_goal_multiagent_rsl_rl.py"
    )
    if not Path(train_script).is_file():
        sys.exit(f"[visualize_observation] Training script not found: {train_script}")

    cmd: list[str] = [
        sys.executable, "-u", train_script,
        "--config", config_path,
        "--test_mode", "obs_visualization",
        "--obs_viz_env_idx", str(known.env_idx),
        "--obs_viz_agent_idx", str(known.agent_idx),
        "--obs_viz_warmup_steps", str(known.num_warmup_steps),
    ]
    if known.warmup_action is not None:
        cmd += ["--obs_viz_warmup_action", *(str(float(v)) for v in known.warmup_action)]
    if known.num_envs is not None:
        cmd += ["--num_envs", str(known.num_envs)]

    # Pass through any extra flags the user supplied (--headless, --device, etc.)
    cmd += passthrough

    print(f"[visualize_observation] Launching: {' '.join(cmd)}", flush=True)
    print(f"[visualize_observation] Output will be saved to: artifacts/obs_debug/", flush=True)

    # Preserve PYTHONPATH so Isaac Lab imports work
    env = os.environ.copy()
    if "PYTHONPATH" not in env:
        env["PYTHONPATH"] = "."
    elif "." not in env["PYTHONPATH"].split(os.pathsep):
        env["PYTHONPATH"] = "." + os.pathsep + env["PYTHONPATH"]

    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
