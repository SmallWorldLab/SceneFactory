#!/usr/bin/env bash
# Weather-EXPOSURE ablation, arm A: ALL DRY.
#
# Trains the model-E recipe (KNN sampler, 256 worlds x 16 agents) on a pool where
# every world is at 0.0 mm water film -> mu = 1.000 uniformly.  The resulting
# policy has never encountered wet ground.
#
# Pairs with run_train_knn_wet0to08.sh.  The two runs differ ONLY in the scene
# pool's water_film_mm; everything else is byte-identical.
#
# Usage:
#   bash run_train_knn_alldry.sh                 # headless, GPU 0
#   GPU=1 bash run_train_knn_alldry.sh           # headless, GPU 1
#   bash run_train_knn_alldry.sh --no-headless   # with GUI
#
# Checkpoints: logs/rsl_rl/waymo_physx_256/<stamp>_knn_16agents_alldry/
# Console log: artifacts/train_knn_alldry/train_<stamp>.log
#   (the previous launcher did NOT redirect stdout, so when a run died with
#    "Failed to get contact force matrix from backend" the traceback was lost
#    with the terminal.  Do not remove the tee.)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG=configs/scene_factory/waymo_physx_256_train_16agents_knn_alldry.yaml
POOL=configs/scene_factory/generated/scene_factory_256scene_0414_alldry.yaml
GPU="${GPU:-0}"
# Interpreter. Defaults to whatever python is active, which is what you want
# after `conda activate scenefactory`. Override with SF_PYTHON if your Isaac
# Sim environment is not the active one.
PY="${SF_PYTHON:-$(command -v python)}"
[ -x "$PY" ] || PY=python

[ -f "$CONFIG" ] || { echo "FATAL: config missing: $CONFIG"; exit 1; }
[ -f "$POOL" ]   || { echo "FATAL: scene pool missing: $POOL"; exit 1; }

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR=artifacts/train_knn_alldry
mkdir -p "$LOGDIR"
LOG="$LOGDIR/train_${STAMP}.log"

{
  echo "=== weather-exposure ablation, arm A: ALL DRY ==="
  date
  echo "config : $CONFIG"
  echo "pool   : $POOL"
  echo "gpu    : $GPU"
  echo "--- diff vs the model-E config (expect exactly 2 lines: pool + run_name) ---"
  diff configs/scene_factory/waymo_physx_256_train_16agents_knn.yaml "$CONFIG" || true
  echo "-------------------------------------------------------------------------"
} 2>&1 | tee "$LOG"

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PY" -u \
  src/train_student_vehicle_goal_multiagent_rsl_rl.py \
  --config "$CONFIG" \
  --headless \
  "$@" 2>&1 | tee -a "$LOG"

echo "exit=${PIPESTATUS[0]}  log=$LOG" | tee -a "$LOG"
