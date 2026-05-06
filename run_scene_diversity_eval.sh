#!/bin/bash
# Scene diversity ablation: eval all 6 checkpoints (8/16/32/64/128/256 unique scenes)
# 64 unseen test scenes, original OD, dry asphalt, invincible, headless
# Runs sequentially on cuda:0

export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"  # cuda:0 within the CUDA_VISIBLE_DEVICES=1 context = physical GPU 1

declare -A CONFIGS
CONFIGS[8]="configs/scene_factory/generated/eval_scenediv_8unique_256total_model_299_test64_dry.yaml"
CONFIGS[16]="configs/scene_factory/generated/eval_scenediv_16unique_256total_model_299_test64_dry.yaml"
CONFIGS[32]="configs/scene_factory/generated/eval_scenediv_32unique_256total_model_300_test64_dry.yaml"
CONFIGS[64]="configs/scene_factory/generated/eval_scenediv_64unique_256total_model_299_test64_dry.yaml"
CONFIGS[128]="configs/scene_factory/generated/eval_scenediv_128unique_256total_model_300_test64_dry.yaml"
CONFIGS[256]="configs/scene_factory/generated/eval_scenediv_256unique_256total_model_300_test64_dry.yaml"

declare -A CKPTS
CKPTS[8]="logs/rsl_rl/scene_factory_goal_reaching_roads/2026-04-28_18-46-44_scene_factory_8unique_256total_random_train_fastgoal_v8_sysid4_noweather/model_299.pt"
CKPTS[16]="logs/rsl_rl/scene_factory_goal_reaching_roads/2026-04-29_01-40-49_scene_factory_16unique_256total_random_train_fastgoal_v8_sysid4_noweather/model_299.pt"
CKPTS[32]="logs/rsl_rl/scene_factory_goal_reaching_roads/2026-04-29_13-27-55_scene_factory_32unique_256total_random_train_fastgoal_v8_sysid4_noweather/model_300.pt"
CKPTS[64]="logs/rsl_rl/scene_factory_goal_reaching_roads/2026-04-29_16-36-31_scene_factory_64unique_256total_random_train_fastgoal_v8_sysid4_noweather/model_299.pt"
CKPTS[128]="logs/rsl_rl/scene_factory_goal_reaching_roads/2026-04-30_08-58-53_scene_factory_128unique_256total_random_train_fastgoal_v8_sysid4_noweather/model_300.pt"
CKPTS[256]="logs/rsl_rl/scene_factory_goal_reaching_roads/2026-04-22_13-34-40_scene_factory_256scene_random_0414_train_fastgoal_v8_sysid4_noweather/model_300.pt"

for N in 8 16 32 64 128 256; do
  echo "========================================"
  echo "Evaluating: ${N} unique scenes"
  echo "========================================"
  PYTHONPATH=. python -u src/train_student_vehicle_goal_multiagent_rsl_rl.py \
    --config "${CONFIGS[$N]}" \
    --test_mode scene_factory_policy_eval \
    --checkpoint_path "${CKPTS[$N]}" \
    --invincible \
    --headless \
    --no-use_fabric \
    --no-video \
    --device "$DEVICE"
  echo "Done: ${N} unique scenes"
done

echo "All scene diversity evals complete."
