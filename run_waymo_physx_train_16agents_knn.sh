#!/usr/bin/env bash
# Phase 1: KNN road-point sampler, unmodified Waymo data, 256 worlds x 16 agents.
# Produces the driving-competent checkpoint for Phase 2 workzone fine-tuning.
#
# Usage:
#   bash run_waymo_physx_train_16agents_knn.sh [--device cuda:1] [extra args...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHONPATH=. python -u \
    src/train_student_vehicle_goal_multiagent_rsl_rl.py \
    --config configs/scene_factory/waymo_physx_256_train_16agents_knn.yaml \
    --headless \
    "$@"
