#!/usr/bin/env bash
# Weather-EXPOSURE ablation, arm B: EVEN 0.0-0.8 mm PRECIPITATION.
#
# Trains the model-E recipe (KNN sampler, 256 worlds x 16 agents) on a pool whose
# water film is spread evenly over [0.0, 0.8] mm.  Effective mu spans 0.804-1.000
# (mean 0.922) with ZERO worlds past the TRFC mu=0 cliff -- unlike the pool E
# itself trained on, which had 73/256 worlds on frictionless ice.
#
# Pairs with run_train_knn_alldry.sh.  The two runs differ ONLY in the scene
# pool's water_film_mm; everything else is byte-identical.
#
# Usage:
#   bash run_train_knn_wet0to08.sh                 # headless, GPU 0
#   GPU=1 bash run_train_knn_wet0to08.sh           # headless, GPU 1
#   bash run_train_knn_wet0to08.sh --no-headless   # with GUI
#
# Checkpoints: logs/rsl_rl/waymo_physx_256/<stamp>_knn_16agents_wet0to08/
# Console log: artifacts/train_knn_wet0to08/train_<stamp>.log
#   (the previous launcher did NOT redirect stdout, so when a run died with
#    "Failed to get contact force matrix from backend" the traceback was lost
#    with the terminal.  Do not remove the tee.)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG=configs/scene_factory/waymo_physx_256_train_16agents_knn_wet0to08.yaml
POOL=configs/scene_factory/generated/scene_factory_256scene_0414_wet0to08.yaml
GPU="${GPU:-0}"
# Interpreter. Defaults to whatever python is active, which is what you want
# after `conda activate scenefactory`. Override with SF_PYTHON if your Isaac
# Sim environment is not the active one.
PY="${SF_PYTHON:-$(command -v python)}"
[ -x "$PY" ] || PY=python

[ -f "$CONFIG" ] || { echo "FATAL: config missing: $CONFIG"; exit 1; }
[ -f "$POOL" ]   || { echo "FATAL: scene pool missing: $POOL"; exit 1; }

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR=artifacts/train_knn_wet0to08
mkdir -p "$LOGDIR"
LOG="$LOGDIR/train_${STAMP}.log"

{
  echo "=== weather-exposure ablation, arm B: EVEN 0.0-0.8 mm ==="
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
