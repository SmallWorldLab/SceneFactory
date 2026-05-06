#!/bin/bash
# ============================================================
#  Physics-blind (v8) vs Physics-aware (v7) evaluation
#  2×2 matrix: (policy) × (surface condition)
#
#  Run A: v7 @ model_600  — dry AC      (in-distribution for v7)
#  Run B: v8 @ model_300  — dry AC      (in-distribution for v8)
#  Run C: v7 @ model_600  — hard SMA 2mm (expected graceful degradation)
#  Run D: v8 @ model_300  — hard SMA 2mm (expected worst case)
#
#  New metrics logged: Metrics/mean_min_ttc_s, Metrics/near_miss_rate,
#                      Metrics/mean_max_drac, Metrics/high_drac_rate
#
#  Usage: bash run_v8_vs_v7_physics_blind_eval.sh [cuda_device]
#  Default device: cuda:0
# ============================================================

set -e

DEVICE="${1:-cuda:0}"
echo "Using device: $DEVICE"

V7_CKPT="logs/rsl_rl/scene_factory_goal_reaching_roads/2026-04-20_22-54-20_scene_factory_256scene_random_0414_train_fastgoal_v7_sysid4_weather/model_600.pt"
# Both v8 conditions use the SAME checkpoint so only physics differs (not the policy)
V8_CKPT="logs/rsl_rl/scene_factory_goal_reaching_roads/2026-04-22_13-34-40_scene_factory_256scene_random_0414_train_fastgoal_v8_sysid4_noweather/model_300.pt"

V7_DRY_CFG="configs/scene_factory/generated/eval_v7_sysid4_weather_model_600_test64_dry.yaml"
V8_DRY_CFG="configs/scene_factory/generated/eval_v8_sysid4_noweather_model_300_test64_dry.yaml"
V7_WET_CFG="configs/scene_factory/generated/eval_v7_sysid4_weather_model_600_test64_hard_sma2mm.yaml"
V8_WET_CFG="configs/scene_factory/generated/eval_v8_sysid4_noweather_model_200_test64_hard_sma2mm.yaml"

# Verify checkpoints and configs exist before starting
for f in "$V7_CKPT" "$V8_CKPT" "$V7_DRY_CFG" "$V8_DRY_CFG" "$V7_WET_CFG" "$V8_WET_CFG"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: missing file: $f"
    exit 1
  fi
done

run_eval() {
  local label="$1"
  local config="$2"
  local ckpt="$3"
  echo ""
  echo "========================================================"
  echo "  Run $label"
  echo "  config:     $config"
  echo "  checkpoint: $ckpt"
  echo "  started:    $(date)"
  echo "========================================================"
  PYTHONPATH=. python -u src/train_student_vehicle_goal_multiagent_rsl_rl.py \
    --config "$config" \
    --test_mode scene_factory_policy_eval \
    --checkpoint_path "$ckpt" \
    --invincible \
    --headless \
    --no-use_fabric \
    --no-video \
    --device "$DEVICE"
  echo "  finished:   $(date)"
}

run_eval "A — v7 (physics-aware)  on DRY AC      [in-distrib]"   "$V7_DRY_CFG" "$V7_CKPT"
run_eval "B — v8 (physics-blind)  on DRY AC      [in-distrib]"   "$V8_DRY_CFG" "$V8_CKPT"
run_eval "C — v7 (physics-aware)  on WET SMA 2mm [OOD physics]"  "$V7_WET_CFG" "$V7_CKPT"
run_eval "D — v8 (physics-blind)  on WET SMA 2mm [OOD physics]"  "$V8_WET_CFG" "$V8_CKPT"

echo ""
echo "========================================================"
echo "All 4 runs complete.  $(date)"
echo "Key metrics to compare across runs:"
echo "  Metrics/success_rate"
echo "  Metrics/collision_rate"
echo "  Metrics/lane_forbidden_done_rate"
echo "  Metrics/mean_min_ttc_s"
echo "  Metrics/near_miss_rate"
echo "  Metrics/mean_max_drac"
echo "  Metrics/high_drac_rate"
echo "========================================================"
