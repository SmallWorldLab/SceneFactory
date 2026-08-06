#!/usr/bin/env bash
# Braking friction sweep: consistent injected entry speed, mu = 1.0 -> 0.0.
# Same PhysX vehicle used to train the workzone (dynamics_mode=physx, sysid v4).
#
# 11 worlds, one mu each (1.0,0.9,...,0.1,0.0). Each: inject v0 (~6 m/s) + wheel
# pre-spin, then FULL BRAKE to a stop. mu=0 never stops (censored at cap).
# Outputs:
#   logs/rsl_rl/brake_friction_sweep/<timestamp>_physx_mu_sweep/
#     brake_sweep_summary.csv / .json
#     brake_sweep_trajectory.csv
#     brake_sweep_distance_vs_mu.png
#
# Pins CUDA_VISIBLE_DEVICES=0 so the workzone optimizer on GPU 1 is untouched.
#
# Usage:
#   bash run_brake_friction_sweep.sh                # headless (default)
#   bash run_brake_friction_sweep.sh --no-headless  # with GUI
#
# Tunables (env vars): BFS_V0_MPS (default 6.0), BFS_MAX_BRAKE_STEPS (1500).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

HEADLESS="--headless"
PASSTHROUGH=()
for arg in "$@"; do
    [ "$arg" = "--no-headless" ] && HEADLESS="" || PASSTHROUGH+=("$arg")
done

echo "[brake-sweep] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

PYTHONPATH=. python -u src/train_student_vehicle_goal_multiagent_rsl_rl.py \
    --config configs/scene_factory/brake_friction_sweep.yaml \
    $HEADLESS \
    --test_mode brake_friction_sweep \
    "${PASSTHROUGH[@]}"
