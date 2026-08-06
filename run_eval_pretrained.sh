#!/usr/bin/env bash
# run_eval_pretrained.sh
# ─────────────────────────────────────────────────────────────────────────────
# Evaluate the two pre-trained policies shipped with this repo. This is the
# quickest way to confirm a working install end to end.
#
#   dry_trained       — trained on an all-dry scene pool
#   weather_exposed   — trained on the same pool with 0-12 mm water films
#
# Each is evaluated on the held-out 199-scene pool under two conditions:
#   dry    — Asphalt Concrete, 0 mm water film
#   wet    — 5 mm water film, which is inside the range where traction binds
#
# For the full result -- 4 surfaces x N seeds, with significance tests -- use
# run_paper_table4_eval.sh instead. This script is one seed per cell.
#
# Expected runtime: ~5 min per condition on a single GPU.
# ─────────────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")"

TRAINER="src/train_student_vehicle_goal_multiagent_rsl_rl.py"
COMMON="--invincible --headless --no-use_fabric --device cuda:0"
CFG=configs/scene_factory/eval_knn_backbone_weather_multiseed.yaml

CKPT_DRY="${CKPT_DRY:-checkpoints/dry_trained_iter900.pt}"
CKPT_WET="${CKPT_WET:-checkpoints/weather_exposed_iter900.pt}"

for f in "$CKPT_DRY" "$CKPT_WET" "$CFG" \
    configs/scene_factory/generated/eval_unseen_199scenes_dry.yaml \
    configs/scene_factory/generated/eval_unseen_199scenes_wet5mm.yaml; do
  [ -f "$f" ] || { echo "ERROR: missing file: $f"; exit 1; }
done

echo "============================================================"
echo " SceneFactory pre-trained policy evaluation"
echo " $(date)"
echo "============================================================"

# The weather token, and why the two policies are not fed the same input.
#
# The dry-trained policy saw water_film_mm = 0.0 in EVERY world, so it has never
# seen a nonzero first element of the weather token. Handing it the live token on
# a wet road is an out-of-distribution INPUT, not a harder road, and it collapses
# for that reason rather than because of traction. So it is evaluated with the
# weather channel blinded in every condition: pinned to the dry-AC constant
# [0,1,0,0], which is bit-identical to the live token on the dry pool.
#
# The weather-exposed policy keeps the live token. Using weather information is
# the capability under test.
i=1
for policy in dry_trained weather_exposed; do
  case "$policy" in
    dry_trained)     CKPT="$CKPT_DRY"; TOKEN="--obs_weather_context_blind" ;;
    weather_exposed) CKPT="$CKPT_WET"; TOKEN="--no-obs_weather_context_blind" ;;
  esac
  for wx in dry wet5mm; do
    echo ""
    echo "[${i}/4] ${policy} — ${wx}"
    PYTHONPATH=. python -u "$TRAINER" $COMMON \
      --config "$CFG" \
      --scene_factory_config "configs/scene_factory/generated/eval_unseen_199scenes_${wx}.yaml" \
      --test_mode scene_factory_policy_eval \
      $TOKEN \
      --checkpoint_path "$CKPT"
    i=$((i + 1))
  done
done

echo ""
echo "============================================================"
echo " Done. Results written to timestamped directories above."
echo "============================================================"
