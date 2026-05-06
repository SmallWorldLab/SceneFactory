#!/bin/bash
# Eval: v8 (PhysX-trained) policy → Bicycle dynamics
# PhysX train, Bicycle eval — reverse sim-to-sim transfer test

PYTHONPATH=. python -u src/train_student_vehicle_goal_multiagent_rsl_rl.py \
  --config configs/scene_factory/generated/eval_v8_physx_to_bicycle_transfer_model_300_test64_dry.yaml \
  --test_mode scene_factory_policy_eval \
  --checkpoint_path logs/rsl_rl/scene_factory_goal_reaching_roads/2026-04-22_13-34-40_scene_factory_256scene_random_0414_train_fastgoal_v8_sysid4_noweather/model_300.pt \
  --dynamics_mode bicycle \
  --invincible \
  --enable_cameras \
  --video \
  --video_view_mode per_env \
  --video_camera_pose_mode top_down \
  --headless \
  --no-use_fabric \
  --device cuda:0
