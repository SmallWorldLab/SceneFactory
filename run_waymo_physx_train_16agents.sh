#!/usr/bin/env bash
# Full-scale training: weather-aware PhysX vehicle, 256 worlds, 16 agents/world.
#
# 256 worlds × 16 agents = 4096 agent slots, road_points_k=250
# Per-world friction randomization (AC/SMA/OGFC, dry to heavy rain)
#
# Usage:
#   bash run_waymo_physx_train_16agents.sh            # headless (recommended)
#   bash run_waymo_physx_train_16agents.sh --no-headless  # with GUI
#
# Logs and checkpoints: logs/rsl_rl/waymo_physx_256/friction_16agents/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHONPATH=. python -u src/train_student_vehicle_goal_multiagent_rsl_rl.py \
  --config configs/scene_factory/waymo_physx_256_train_16agents.yaml \
  --headless \
  "$@"
