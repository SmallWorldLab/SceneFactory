#!/usr/bin/env bash
# run_eval_pretrained.sh
# ─────────────────────────────────────────────────────────────────────────────
# Evaluate the two pre-trained checkpoints shipped with this repo.
#
#   v7  — weather-aware policy (trained with friction token observation)
#   v8  — no-weather baseline  (friction token masked, same architecture)
#
# Each is evaluated on two conditions:
#   dry  — Asphalt Concrete, 0 mm water film  (mu ≈ 1.105)
#   wet  — SMA surface,     2.0 mm water film  (mu ≈ 0.001, near-hydroplaning)
#
# Results are written to a timestamped directory under the repo root.
# Expected runtime: ~5 min per condition on a single GPU.
# ─────────────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")"

PYTHON="src/train_student_vehicle_goal_multiagent_rsl_rl.py"
COMMON="--invincible --headless --no-use_fabric --device cuda:0"

V7_CKPT="checkpoints/v7_weather_aware_iter600.pt"
V8_CKPT="checkpoints/v8_no_weather_iter300.pt"

V8_WET_CFG="configs/scene_factory/generated/eval_v8_sysid4_noweather_model_200_test64_hard_sma2mm.yaml"

for f in "$V7_CKPT" "$V8_CKPT" \
    "configs/scene_factory/generated/eval_v7_sysid4_weather_model_600_test64_dry.yaml" \
    "configs/scene_factory/generated/eval_v8_sysid4_noweather_model_300_test64_dry.yaml" \
    "configs/scene_factory/generated/eval_v7_sysid4_weather_model_600_test64_hard_sma2mm.yaml" \
    "$V8_WET_CFG"; do
  [ -f "$f" ] || { echo "ERROR: missing file: $f"; exit 1; }
done

echo "============================================================"
echo " SceneFactory pre-trained policy evaluation"
echo " $(date)"
echo "============================================================"

# ── v7: weather-aware, dry ──────────────────────────────────────────────────
echo ""
echo "[1/4] v7 weather-aware — DRY (AC, 0 mm)"
PYTHONPATH=. python -u $PYTHON $COMMON \
  --config configs/scene_factory/generated/eval_v7_sysid4_weather_model_600_test64_dry.yaml \
  --test_mode scene_factory_policy_eval \
  --checkpoint_path "$V7_CKPT"

# ── v8: no-weather baseline, dry ────────────────────────────────────────────
echo ""
echo "[2/4] v8 no-weather baseline — DRY (AC, 0 mm)"
PYTHONPATH=. python -u $PYTHON $COMMON \
  --config configs/scene_factory/generated/eval_v8_sysid4_noweather_model_300_test64_dry.yaml \
  --test_mode scene_factory_policy_eval \
  --checkpoint_path "$V8_CKPT"

# ── v7: weather-aware, heavy wet ────────────────────────────────────────────
echo ""
echo "[3/4] v7 weather-aware — HEAVY WET (SMA, 2.0 mm, mu≈0.001)"
PYTHONPATH=. python -u $PYTHON $COMMON \
  --config configs/scene_factory/generated/eval_v7_sysid4_weather_model_600_test64_hard_sma2mm.yaml \
  --test_mode scene_factory_policy_eval \
  --checkpoint_path "$V7_CKPT"

# ── v8: no-weather baseline, heavy wet ──────────────────────────────────────
echo ""
echo "[4/4] v8 no-weather baseline — HEAVY WET (SMA, 2.0 mm, mu≈0.001)"
PYTHONPATH=. python -u $PYTHON $COMMON \
  --config "$V8_WET_CFG" \
  --test_mode scene_factory_policy_eval \
  --checkpoint_path "$V8_CKPT"

echo ""
echo "============================================================"
echo " Done. Results written to timestamped directories above."
echo "============================================================"
