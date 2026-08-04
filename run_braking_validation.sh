#!/usr/bin/env bash
# NHTSA-style braking validation of the PhysX articulated vehicle.
# Same vehicle used to train the workzone (dynamics_mode=physx, sysid v4).
#
# 9 worlds, one mu each: 0.30 0.40 0.50 0.60 0.66 0.70 0.86 1.00 1.10
# Each world throttles up to v0=27.78 m/s (100 km/h) then FULL BRAKE to a stop.
# Output CSV: logs/rsl_rl/braking_validation/<timestamp>_physx_nhtsa/braking_validation.csv
#
# Pins CUDA_VISIBLE_DEVICES=0 so the workzone optimizer on GPU 1 is untouched.
#
# Usage:
#   bash run_braking_validation.sh                # headless (default)
#   bash run_braking_validation.sh --no-headless  # with GUI

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

HEADLESS="--headless"
PASSTHROUGH=()
for arg in "$@"; do
    [ "$arg" = "--no-headless" ] && HEADLESS="" || PASSTHROUGH+=("$arg")
done

echo "[braking-validation] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

PYTHONPATH=. python -u src/train_student_vehicle_goal_multiagent_rsl_rl.py \
    --config configs/scene_factory/braking_validation.yaml \
    $HEADLESS \
    --test_mode braking_validation \
    "${PASSTHROUGH[@]}"
