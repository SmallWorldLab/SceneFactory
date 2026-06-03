#!/usr/bin/env bash
# Ablation: train the workzone policy using PHYSX dynamics (no bicycle model).
#
# Config is identical to the bicycle run EXCEPT:
#   - dynamics_mode: PhysX (default articulation — no bicycle fields)
#   - choco_idle_speed_threshold_mps: 1.0  (same as bicycle, not the old 2.5)
#   - learning_rate: 3.0e-4                (same as bicycle)
#
# Purpose: isolate whether bicycle dynamics is necessary, or if the reward
# threshold (2.5 m/s idle threshold vs PhysX ~1.5 m/s ceiling) was the
# real bottleneck in the earlier PhysX run 2.
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
#   bash run_workzone_train_physx.sh                # headless, from scratch
#   bash run_workzone_train_physx.sh --no-headless  # with GUI

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CHECKPOINT="${CHECKPOINT:-}"

echo "=== Workzone training (PhysX ablation) ==="
if [ -n "$CHECKPOINT" ]; then
    if [ ! -f "$CHECKPOINT" ]; then
        echo "ERROR: Checkpoint not found: $CHECKPOINT"
        exit 1
    fi
    echo "  Base checkpoint : $CHECKPOINT"
else
    echo "  Base checkpoint : (none — training from scratch)"
fi
echo "  Config          : configs/scene_factory/workzone_train_physx_ablation.yaml"
echo "  Dynamics        : PhysX (no bicycle model)"
echo "  Idle threshold  : 1.0 m/s (matches bicycle run)"
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
    --config configs/scene_factory/workzone_train_physx_ablation.yaml \
    $HEADLESS_FLAG \
    $RESUME_ARGS \
    "$@"
