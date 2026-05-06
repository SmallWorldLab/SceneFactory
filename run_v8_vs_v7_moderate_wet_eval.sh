#!/usr/bin/env bash
# run_v8_vs_v7_moderate_wet_eval.sh
# ─────────────────────────────────────────────────────────────────────────────
# 2-condition eval: v7 vs v8 on MODERATE wet road (AC + 0.5mm water film, mu~=0.859)
# This complements run_v8_vs_v7_physics_blind_eval.sh (which used SMA 2mm / mu~=1e-3).
#
# Dry results (already collected):
#   v7_dry: success=77.2%   v8_dry: success=77.9%  (effectively equal)
# Hard-wet results (already collected):
#   v7_wet: success= 8.2%   v8_wet: success= 8.0%  (both collapse — physics dominates)
# Expected here (moderate wet — vehicles remain controllable):
#   v7 should retain higher success and lower near-miss rates than v8.
#
# Physics: AC 0.5mm → mu_static ~= 0.859 (~22% reduction vs dry AC 1.105)
# ─────────────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")"

PYTHON="src/train_student_vehicle_goal_multiagent_rsl_rl.py"
COMMON="--invincible --headless --no-use_fabric --device cuda:0"

V7_CKPT="logs/rsl_rl/scene_factory_goal_reaching_roads/2026-04-20_22-54-20_scene_factory_256scene_random_0414_train_fastgoal_v7_sysid4_weather/model_600.pt"
V8_CKPT="logs/rsl_rl/scene_factory_goal_reaching_roads/2026-04-22_13-34-40_scene_factory_256scene_random_0414_train_fastgoal_v8_sysid4_noweather/model_300.pt"

# Validate files exist
for f in "$V7_CKPT" "$V8_CKPT" \
    "configs/scene_factory/generated/eval_v7_sysid4_weather_model_600_test64_moderate_ac0p5mm.yaml" \
    "configs/scene_factory/generated/eval_v8_sysid4_noweather_model_300_test64_moderate_ac0p5mm.yaml"; do
  [ -f "$f" ] || { echo "ERROR: missing file: $f"; exit 1; }
done

echo "============================================================"
echo " v7 vs v8 MODERATE-WET eval  (AC 0.5mm, mu~=0.859)"
echo " $(date)"
echo "============================================================"

# ── Condition C-mod: v7 + moderate wet ──────────────────────────────────────
echo ""
echo "[$(date +%H:%M:%S)] C-mod: v7 (friction-aware) on moderate wet (AC 0.5mm)"
PYTHONPATH=. python -u $PYTHON $COMMON \
  --config configs/scene_factory/generated/eval_v7_sysid4_weather_model_600_test64_moderate_ac0p5mm.yaml \
  --test_mode scene_factory_policy_eval \
  --checkpoint_path "$V7_CKPT"

# ── Condition D-mod: v8 + moderate wet ──────────────────────────────────────
echo ""
echo "[$(date +%H:%M:%S)] D-mod: v8 (physics-blind) on moderate wet (AC 0.5mm)"
PYTHONPATH=. python -u $PYTHON $COMMON \
  --config configs/scene_factory/generated/eval_v8_sysid4_noweather_model_300_test64_moderate_ac0p5mm.yaml \
  --test_mode scene_factory_policy_eval \
  --checkpoint_path "$V8_CKPT"

echo ""
echo "============================================================"
echo " All moderate-wet evals complete  $(date)"
echo "============================================================"
echo ""
echo "Results in: logs/rsl_rl/scene_factory_goal_reaching_eval/"
echo "  eval_v7_sysid4_weather_model_600_test64_moderate_ac0p5mm/"
echo "  eval_v8_sysid4_noweather_model_300_test64_moderate_ac0p5mm/"
echo ""
echo "Key metrics to compare (run scripts/summarize_2x2_eval.py for full table):"
echo "  success_rate, collision_rate, mean_min_ttc_s, near_miss_rate, high_drac_rate"
