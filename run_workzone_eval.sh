#!/usr/bin/env bash
# Evaluate the workzone bicycle policy on a random workzone scene with video capture.
#
# Loads the latest bicycle checkpoint and runs one deterministic episode per world,
# recording a top-down BEV video.
#
# Usage:
#   bash run_workzone_eval.sh                          # use default checkpoint
#   bash run_workzone_eval.sh --no-headless            # with GUI (no video)
#   CHECKPOINT=path/to/model.pt bash run_workzone_eval.sh  # custom checkpoint

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default: latest bicycle training checkpoint
CHECKPOINT="${CHECKPOINT:-logs/rsl_rl/workzone_optimization/2026-06-02_09-21-06_workzone_train/model_100.pt}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "  Set CHECKPOINT=path/to/model.pt to override."
    exit 1
fi

echo "=== Workzone policy evaluation ==="
echo "  Checkpoint : $CHECKPOINT"
echo "  Config     : configs/scene_factory/workzone_train.yaml"
echo "  Scene dir  : data/processed/workzone_scenes_json/"
echo "  Video      : artifacts/workzone_eval_video/"
echo ""

HEADLESS_FLAG="--headless"
for arg in "$@"; do
    [ "$arg" = "--no-headless" ] && HEADLESS_FLAG="" && break
done

PYTHONPATH=. python -u src/train_student_vehicle_goal_multiagent_rsl_rl.py \
    --config configs/scene_factory/workzone_train.yaml \
    --test_mode scene_factory_policy_eval \
    --checkpoint_path "$CHECKPOINT" \
    $HEADLESS_FLAG \
    --no-use_fabric \
    --invincible \
    --device cuda:0 \
    --num_envs 4 \
    --dynamics_mode bicycle \
    --enable_cameras \
    --video \
    --video_view_mode per_env \
    --video_width 1920 \
    --video_height 1080 \
    --video_fps 30 \
    --video_step_stride 1 \
    --video_vehicle_proxy_markers \
    --video_vehicle_proxy_z_offset_m 0.35 \
    --video_camera_pose_mode top_down \
    --video_name_prefix workzone_eval \
    --log_dir artifacts/workzone_eval_video \
    "$@"
