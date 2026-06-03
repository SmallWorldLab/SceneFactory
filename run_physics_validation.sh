#!/usr/bin/env bash
# Physics validation test — PhysX vehicle dynamics, no Waymo data needed.
#
# Tests 3 things across 14 parallel worlds (1 agent each):
#   Phase 1 (envs 0–3):  throttle sweep [0.25, 0.50, 0.75, 1.00]  — must accelerate, monotonically
#   Phase 2 (envs 4–8):  steer sweep [-1, -0.5, 0, 0.5, 1]        — must turn, symmetric L/R
#   Phase 3 (envs 9–13): friction μ = [0.02, 0.20, 0.40, 0.65, 0.95] — must affect dynamics
#
# Expected runtime: ~3–5 min (setting up Isaac Sim + 648 env steps)
#
# Output: logs/rsl_rl/physics_validation/<timestamp>_physx_test/physics_validation_report.json
#
# Usage:
#   bash run_physics_validation.sh              # headless (default)
#   bash run_physics_validation.sh --no-headless  # with GUI

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HEADLESS="--headless"
for arg in "$@"; do
    [ "$arg" = "--no-headless" ] && HEADLESS="" && break
done

PYTHONPATH=. python -u src/train_student_vehicle_goal_multiagent_rsl_rl.py \
    --config configs/scene_factory/physics_validation.yaml \
    $HEADLESS \
    --test_mode physics_validation \
    "$@"
