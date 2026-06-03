#!/usr/bin/env bash
# Train the workzone safety policy using kinematic BICYCLE dynamics.
#
# Uses dynamics_mode=bicycle (bypasses PhysX torque plateau).
# Config: configs/scene_factory/workzone_train.yaml
#
# By default trains from scratch (no checkpoint). To fine-tune from a pretrained
# checkpoint, set the CHECKPOINT env var:
#   CHECKPOINT=checkpoints/v8_no_weather_iter300.pt bash run_workzone_train_bicycle.sh
#
# Pre-requisites:
#   1. Generate workzone training scenes (already done if data/ is populated):
#        python -m src.workzone_scene_generator \
#          --output_dir data/processed/workzone_scenes_json \
#          --num_worlds 32 --seed 42
#
# Logs and checkpoints → logs/rsl_rl/workzone_optimization/workzone_train/
#
# Usage:
#   bash run_workzone_train_bicycle.sh                     # headless, from scratch
#   bash run_workzone_train_bicycle.sh --no-headless       # with GUI
#   CHECKPOINT=checkpoints/my_policy.pt bash run_workzone_train_bicycle.sh  # from checkpoint

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CHECKPOINT="${CHECKPOINT:-}"

echo "=== Workzone training ==="
if [ -n "$CHECKPOINT" ]; then
    if [ ! -f "$CHECKPOINT" ]; then
        echo "ERROR: Checkpoint not found: $CHECKPOINT"
        exit 1
    fi
    echo "  Base checkpoint : $CHECKPOINT"
else
    echo "  Base checkpoint : (none — training from scratch)"
fi
echo "  Config          : configs/scene_factory/workzone_train.yaml"
echo "  Scene dir       : data/processed/workzone_scenes_json/"
echo ""

HEADLESS_FLAG="--headless"
for arg in "$@"; do
    [ "$arg" = "--no-headless" ] && HEADLESS_FLAG="" && break
done

RESUME_ARGS=""
if [ -n "$CHECKPOINT" ]; then
    RESUME_ARGS="--resume_from $CHECKPOINT"
fi

PYTHONPATH=. python -u src/train_student_vehicle_goal_multiagent_rsl_rl.py \
    --config configs/scene_factory/workzone_train.yaml \
    $HEADLESS_FLAG \
    $RESUME_ARGS \
    "$@"
