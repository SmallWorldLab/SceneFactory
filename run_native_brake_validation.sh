#!/usr/bin/env bash
# Native-speed brake validation: dry vs wet, one car, full trajectory.
# Same PhysX vehicle used to train the workzone (dynamics_mode=physx, sysid v4).
#
# world 0 = DRY (mu 0.90), world 1 = WET (mu 0.50).
# Each: accelerate to ~6 m/s (native throttle plateau) then FULL BRAKE to a stop.
# Records per-step trajectory CSV per condition + a summary + a comparison plot:
#   logs/rsl_rl/native_brake_validation/<timestamp>_physx_dry_wet/
#     native_brake_dry_trajectory.csv
#     native_brake_wet_trajectory.csv
#     native_brake_summary.csv / .json
#     native_brake_compare.png
#
# Pins CUDA_VISIBLE_DEVICES=0 so the workzone optimizer on GPU 1 is untouched.
#
# Usage:
#   bash run_native_brake_validation.sh                # headless (default)
#   bash run_native_brake_validation.sh --no-headless  # with GUI
#
# Tunables (env vars): NBV_TARGET_MPS (default 6.0), NBV_MAX_DRIVE_STEPS (1200).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

HEADLESS="--headless"
PASSTHROUGH=()
for arg in "$@"; do
    [ "$arg" = "--no-headless" ] && HEADLESS="" || PASSTHROUGH+=("$arg")
done

echo "[native-brake] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

PYTHONPATH=. python -u src/train_student_vehicle_goal_multiagent_rsl_rl.py \
    --config configs/scene_factory/native_brake_validation.yaml \
    $HEADLESS \
    --test_mode native_brake_validation \
    "${PASSTHROUGH[@]}"
