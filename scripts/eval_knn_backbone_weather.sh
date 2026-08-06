#!/usr/bin/env bash
# Weather ablation of the KNN workzone backbone on the held-out 199-scene pool.
#
# Policy under test (single checkpoint, no second model trained):
#   logs/rsl_rl/waymo_physx_256/2026-06-23_23-51-35_knn_16agents/model_2999.pt
#
# Design: 3 weather x 2 observation modes x N seeds.
#   weather : dry (AC 0.0mm, mu 1.000 after cap)
#             w04 (AC 0.4mm, mu 0.904)
#             w08 (AC 0.8mm, mu 0.804)   <-- CEILING, TRFC cliffs to 0 at 0.85mm
#   obs     : cond  (--no-obs_weather_context_blind) weather token is live
#             blind (--obs_weather_context_blind)    token pinned to dry-AC
#
# Protocol held fixed at terminating + dataset spawns (tightest across-seed
# spread, +/-1.0 on SR, so best power to detect a small effect).
#
# The usable friction span here is only 1.000 -> 0.804 (a 20% band), because the
# wheel-side "min" combine caps mu at 1.0 and TRFC is unusable above 0.8 mm.
# A null result is the expected outcome and is itself the finding: it bounds how
# much the Fig.6 fix in tasks/fix_trfc_yr_hydroplaning_cliff.md would buy.
#
# Usage:
#   bash scripts/eval_knn_backbone_weather.sh [NUM_SEEDS] [GPU]
#     NUM_SEEDS default 5, GPU default 1.
#
# Aggregate: python scripts/aggregate_knn_backbone_eval.py --weather

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

NUM_SEEDS="${1:-5}"
GPU="${2:-1}"
# Interpreter. Defaults to whatever python is active, which is what you want
# after `conda activate scenefactory`. Override with SF_PYTHON if your Isaac
# Sim environment is not the active one.
PY="${SF_PYTHON:-$(command -v python)}"
CKPT=logs/rsl_rl/waymo_physx_256/2026-06-23_23-51-35_knn_16agents/model_2999.pt
CFG=configs/scene_factory/eval_knn_backbone_weather_multiseed.yaml
LOGDIR=artifacts/knn_backbone_weather_eval
mkdir -p "$LOGDIR"

[ -f "$CKPT" ] || { echo "FATAL: checkpoint missing: $CKPT"; exit 1; }
[ -f "$CFG" ]  || { echo "FATAL: config missing: $CFG"; exit 1; }

echo "=== weather ablation: 3 weather x 2 obs x ${NUM_SEEDS} seeds on GPU ${GPU} ==="
date

for wx in dry w04 w08; do
  case "$wx" in
    dry) POOL=configs/scene_factory/generated/eval_unseen_199scenes_dry.yaml ;;
    w04) POOL=configs/scene_factory/generated/eval_unseen_199scenes_ac0p4mm.yaml ;;
    w08) POOL=configs/scene_factory/generated/eval_unseen_199scenes_ac0p8mm.yaml ;;
  esac
  [ -f "$POOL" ] || { echo "FATAL: pool missing: $POOL"; exit 1; }
  for obs in cond blind; do
    case "$obs" in
      cond)  OBS_FLAG="--no-obs_weather_context_blind" ;;
      blind) OBS_FLAG="--obs_weather_context_blind" ;;
    esac
    for s in $(seq 1 "$NUM_SEEDS"); do
      seed=$((100 + s))
      tag="${wx}_${obs}_seed${seed}"
      echo "--- ${tag} ---"
      start=$(date +%s)
      CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PY" -u \
        src/train_student_vehicle_goal_multiagent_rsl_rl.py \
        --config "$CFG" \
        --scene_factory_config "$POOL" \
        --test_mode scene_factory_policy_eval \
        --checkpoint_path "$CKPT" \
        --headless --no-use_fabric \
        --no-invincible --no-random_od \
        $OBS_FLAG \
        --seed "$seed" \
        --scene_factory_random_world_seed "$seed" \
        --run_name "wx_${tag}" \
        > "$LOGDIR/${tag}.log" 2>&1
      echo "    exit=$?  elapsed=$(( $(date +%s) - start ))s"
    done
  done
done

echo "=== done ==="; date
