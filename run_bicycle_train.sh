#!/usr/bin/env bash
# Train bicycle-dynamics ablation (v1) — mirrors v8 with kinematic bicycle model
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHONPATH=. python -u src/train_student_vehicle_goal_multiagent_rsl_rl.py \
  --config configs/scene_factory/generated/scene_factory_256scene_random_train_fastgoal_bicycle_v1.yaml \
  --headless \
  "$@"
