#!/usr/bin/env bash
# Demo training: weather-aware PhysX vehicle on 32 Waymo worlds.
#
# This is the recommended starting point for new users. It exercises the full
# SceneFactory feature stack (sysid-calibrated articulated PhysX vehicle,
# weather-to-friction module, real Waymo road geometry) at a scale that fits
# on a single GPU with ~8–10 GB VRAM.
#
# Scale vs. paper:  32 worlds × 4 agents = 128 slots  (paper: 256 × 16 = 4096)
# Weather:          4 conditions — dry AC / light / moderate / heavy rain SMA
# Expected time:    ~30–60 min to 500 iterations on an RTX 3090 / A100
#
# Usage:
#   bash run_demo_train.sh            # headless (recommended)
#   bash run_demo_train.sh --no-headless   # with GUI (needs display)
#
# Logs and checkpoints are saved to logs/rsl_rl/scene_factory_demo/demo_weather_physx/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHONPATH=. python -u src/train_student_vehicle_goal_multiagent_rsl_rl.py \
  --config configs/scene_factory/demo_weather_physx_train.yaml \
  --headless \
  "$@"
