#!/usr/bin/env bash
# run_v8_vs_v7_heavy_wet_eval.sh
# ─────────────────────────────────────────────────────────────────────────────
# 2-condition eval: v7 vs v8 on HEAVY wet road (AC + 0.8mm water film, mu~=0.804)
# This is more slippery than the moderate eval (AC 0.5mm, mu=0.859) but still
# below the hydroplaning cliff (AC hits floor at 0.9mm).
#
# Previously collected results:
#   Dry:           v7=77.2%  v8=77.9%  DRAC: v7=37.1  v8=50.2
#   Moderate wet:  v7=77.9%  v8=78.0%  DRAC: v7=27.8  v8=58.7  (2.1x gap!)
#   Hard wet:      v7= 8.2%  v8= 8.0%  (both collapse — physics dominates)
# Expected here (heavy wet — ~27% friction reduction, still controllable):
#   DRAC gap between v7 and v8 should widen further.
#
# Physics: AC 0.8mm → mu_static ~= 0.804 (~27% reduction vs dry AC 1.105)
# ─────────────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")"

PYTHON="src/train_student_vehicle_goal_multiagent_rsl_rl.py"
COMMON="--invincible --headless --no-use_fabric --device cuda:0"

V7_CKPT="logs/rsl_rl/scene_factory_goal_reaching_roads/2026-04-20_22-54-20_scene_factory_256scene_random_0414_train_fastgoal_v7_sysid4_weather/model_600.pt"
V8_CKPT="logs/rsl_rl/scene_factory_goal_reaching_roads/2026-04-22_13-34-40_scene_factory_256scene_random_0414_train_fastgoal_v8_sysid4_noweather/model_300.pt"

# Validate files exist
for f in "$V7_CKPT" "$V8_CKPT" \
    "configs/scene_factory/generated/eval_v7_sysid4_weather_model_600_test64_heavy_ac0p8mm.yaml" \
    "configs/scene_factory/generated/eval_v8_sysid4_noweather_model_300_test64_heavy_ac0p8mm.yaml"; do
  [ -f "$f" ] || { echo "ERROR: missing file: $f"; exit 1; }
done

echo "============================================================"
echo " v7 vs v8 HEAVY-WET eval  (AC 0.8mm, mu~=0.804)"
echo " $(date)"
echo "============================================================"

# ── Condition C-heavy: v7 + heavy wet ───────────────────────────────────────
echo ""
echo "[$(date +%H:%M:%S)] C-heavy: v7 (friction-aware) on heavy wet (AC 0.8mm)"
PYTHONPATH=. python -u $PYTHON $COMMON \
  --config configs/scene_factory/generated/eval_v7_sysid4_weather_model_600_test64_heavy_ac0p8mm.yaml \
  --test_mode scene_factory_policy_eval \
  --checkpoint_path "$V7_CKPT"

# ── Condition D-heavy: v8 + heavy wet ───────────────────────────────────────
echo ""
echo "[$(date +%H:%M:%S)] D-heavy: v8 (physics-blind) on heavy wet (AC 0.8mm)"
PYTHONPATH=. python -u $PYTHON $COMMON \
  --config configs/scene_factory/generated/eval_v8_sysid4_noweather_model_300_test64_heavy_ac0p8mm.yaml \
  --test_mode scene_factory_policy_eval \
  --checkpoint_path "$V8_CKPT"

echo ""
echo "============================================================"
echo " All heavy-wet evals complete  $(date)"
echo "============================================================"
echo ""
echo "Results in: logs/rsl_rl/scene_factory_goal_reaching_eval/"
echo "  eval_v7_sysid4_weather_model_600_test64_heavy_ac0p8mm/"
echo "  eval_v8_sysid4_noweather_model_300_test64_heavy_ac0p8mm/"
echo ""
echo "Key metrics to compare:"
echo "  success_rate, collision_rate, mean_min_ttc_s, near_miss_rate, mean_max_drac, high_drac_rate"
