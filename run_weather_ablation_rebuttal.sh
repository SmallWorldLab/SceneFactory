#!/usr/bin/env bash
# Weather-exposure ablation: two training arms x N seeds.
#
# This trains the policies that run_paper_table4_eval.sh evaluates.  The arms differ in
# exactly two config lines (scene pool + run name); the pools differ in exactly
# one field (water_film_mm).  Verify any time with:
#   diff <(grep -v '^\s*#' configs/scene_factory/waymo_physx_256_train_16agents_knn_alldry.yaml) \
#        <(grep -v '^\s*#' configs/scene_factory/waymo_physx_256_train_16agents_knn_wet0to12.yaml)
#
#   arm      pool                mu span         mean    mu<=0.78
#   dry      ..._0414_alldry     1.174 - 1.200   1.179     0/256
#   wet      ..._0414_wet0to12   0.408 - 1.200   0.569   230/256
#
# --seed varies the TRAINING seed only.  scene_factory.random_world_seed stays at
# 42 in both configs, so the scene<->world and scene<->water-film assignments are
# identical across every run.  The only thing a seed changes is PPO/init noise --
# which is precisely the confound reviewers asked us to quantify.
#
# 1500 iterations, ~13.5 h/run (the E backbone plateaus at ~1500; see the config
# header).  Two GPUs -> one arm-pair per ~13.5 h batch.
#
# Usage:
#   bash run_weather_ablation_rebuttal.sh <dry|wet> <seed> <gpu>
#
# Checkpoints: logs/rsl_rl/waymo_physx_256/<stamp>_knn_16agents_<arm>_s<seed>/
# Console log: artifacts/weather_ablation_rebuttal/<arm>_s<seed>_<stamp>.log

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ARM="${1:-}"
SEED="${2:-}"
GPU="${3:-0}"

case "$ARM" in
  dry) CONFIG=configs/scene_factory/waymo_physx_256_train_16agents_knn_alldry.yaml
       POOL=configs/scene_factory/generated/scene_factory_256scene_0414_alldry.yaml
       BASE_RUN=knn_16agents_alldry ;;
  wet) CONFIG=configs/scene_factory/waymo_physx_256_train_16agents_knn_wet0to12.yaml
       POOL=configs/scene_factory/generated/scene_factory_256scene_0414_wet0to12.yaml
       BASE_RUN=knn_16agents_wet0to12 ;;
  *)   echo "usage: bash $0 <dry|wet> <seed> <gpu>"; exit 2 ;;
esac

[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "FATAL: seed must be an integer, got '${SEED}'"; exit 2; }

# Interpreter. Defaults to whatever python is active, which is what you want
# after `conda activate scenefactory`. Override with SF_PYTHON if your Isaac
# Sim environment is not the active one.
PY="${SF_PYTHON:-$(command -v python)}"
[ -x "$PY" ] || PY=python

[ -f "$CONFIG" ] || { echo "FATAL: config missing: $CONFIG"; exit 1; }
[ -f "$POOL" ]   || { echo "FATAL: scene pool missing: $POOL"; exit 1; }

RUN_NAME="${BASE_RUN}_s${SEED}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR=artifacts/weather_ablation_rebuttal
mkdir -p "$LOGDIR"
LOG="$LOGDIR/${ARM}_s${SEED}_${STAMP}.log"

{
  echo "=== weather-exposure ablation (rebuttal rerun) ==="
  date
  echo "arm      : $ARM"
  echo "config   : $CONFIG"
  echo "pool     : $POOL"
  echo "seed     : $SEED   (training seed only; world assignment fixed at 42)"
  echo "run_name : $RUN_NAME"
  echo "gpu      : $GPU"
  echo "iters    : 1500"
} 2>&1 | tee "$LOG"

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PY" -u \
  src/train_student_vehicle_goal_multiagent_rsl_rl.py \
  --config "$CONFIG" \
  --seed "$SEED" \
  --run_name "$RUN_NAME" \
  --max_iterations 1500 \
  --headless \
  "${@:4}" 2>&1 | tee -a "$LOG"

echo "exit=${PIPESTATUS[0]}  log=$LOG" | tee -a "$LOG"
