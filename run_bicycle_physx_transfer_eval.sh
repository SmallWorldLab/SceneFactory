#!/usr/bin/env bash
# Sim-to-sim transfer eval: bicycle-trained policy (model_375) → PhysX dynamics
# Quantifies the physics gap between kinematic bicycle training and rigid-body eval.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CHECKPOINT="logs/rsl_rl/scene_factory_goal_reaching_roads/2026-04-26_02-51-33_scene_factory_256scene_random_train_fastgoal_bicycle_v1/model_375.pt"
CONFIG="configs/scene_factory/generated/eval_bicycle_v1_physx_transfer_model_375_test64_dry.yaml"

PYTHONPATH=. python -u src/train_student_vehicle_goal_multiagent_rsl_rl.py \
  --config "$CONFIG" \
  --test_mode scene_factory_policy_eval \
  --checkpoint_path "$CHECKPOINT" \
  --headless \
  --no-use_fabric \
  --invincible \
  --device cuda:0 \
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
  "$@"
