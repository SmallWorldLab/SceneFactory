#!/usr/bin/env bash
# SceneFactory retrain — DRY-ONLY, 256 worlds x 16 agents, fixed-physics stack.
#
# 256 worlds x 16 agents = 4096 agent slots, road_points_k=250, road_running.
# All worlds AC / clear / 0.0 mm water film (no weather randomization).
# Per-env 1000 m ground cuboids at env_spacing=1500 m -> no overlap.
#
# Usage:
#   bash run_waymo_physx_train_16agents_dry.sh                 # headless, cuda:0
#   bash run_waymo_physx_train_16agents_dry.sh --device cuda:1 # pin to GPU 1
#   bash run_waymo_physx_train_16agents_dry.sh --no-headless   # with GUI
#
# Logs and checkpoints: logs/rsl_rl/waymo_physx_256/dry_16agents_1500spacing/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHONPATH=. python -u src/train_student_vehicle_goal_multiagent_rsl_rl.py \
  --config configs/scene_factory/waymo_physx_256_train_16agents_dry.yaml \
  --headless \
  "$@"
