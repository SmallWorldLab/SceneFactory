#!/usr/bin/env bash
# Multi-seed rebuttal eval of the KNN workzone backbone on the held-out 199-scene pool.
#
# Policy under test (the Phase-1 backbone all workzone_finetune runs load):
#   logs/rsl_rl/waymo_physx_256/2026-06-23_23-51-35_knn_16agents/model_2999.pt
#
# Design: 2 x 2 protocol cross, N seeds each.
#   termination : terminating (crash ends episode) | invincible (June protocol)
#   OD          : spawns (fixed Waymo starts)      | randomod (resampled 20-60 m)
# Each run draws a random 64 of the 199 held-out scenes via
# --scene_factory_random_world_seed, so the across-seed spread is map-sampling
# variance -- exactly the "random unseen test maps" quantity we want to report.
#
# Usage:
#   bash scripts/eval_knn_backbone_multiseed.sh [NUM_SEEDS] [GPU]
#     NUM_SEEDS default 5, GPU default 1 (GPU 0 is running training).
#
# Results: logs/rsl_rl/knn_backbone_rebuttal_eval/*/scene_factory_policy_eval_worlds.jsonl
# Aggregate with: python scripts/aggregate_knn_backbone_eval.py

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
CFG=configs/scene_factory/eval_knn_backbone_unseen_multiseed.yaml
LOGDIR=artifacts/knn_backbone_rebuttal_eval
mkdir -p "$LOGDIR"

[ -f "$CKPT" ] || { echo "FATAL: checkpoint missing: $CKPT"; exit 1; }

echo "=== multi-seed backbone eval: ${NUM_SEEDS} seeds x 4 conditions on GPU ${GPU} ==="
date

for term in terminating invincible; do
  case "$term" in
    terminating) TERM_FLAG="--no-invincible" ;;
    invincible)  TERM_FLAG="--invincible" ;;
  esac
  for od in spawns randomod; do
    case "$od" in
      spawns)   OD_FLAG="--no-random_od" ;;
      randomod) OD_FLAG="--random_od" ;;
    esac
    for s in $(seq 1 "$NUM_SEEDS"); do
      seed=$((100 + s))
      tag="${term}_${od}_seed${seed}"
      out="$LOGDIR/${tag}.log"
      echo "--- ${tag} ---"
      start=$(date +%s)
      CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PY" -u \
        src/train_student_vehicle_goal_multiagent_rsl_rl.py \
        --config "$CFG" \
        --test_mode scene_factory_policy_eval \
        --checkpoint_path "$CKPT" \
        --headless --no-use_fabric \
        $TERM_FLAG $OD_FLAG \
        --seed "$seed" \
        --scene_factory_random_world_seed "$seed" \
        --run_name "rebut_${tag}" \
        > "$out" 2>&1
      rc=$?
      echo "    exit=${rc}  elapsed=$(( $(date +%s) - start ))s  log=${out}"
    done
  done
done

echo "=== done ==="; date
