#!/usr/bin/env bash
# run_workzone_optimize.sh
#
# Outer CEM optimization loop for workzone construction-zone configuration.
#
# Sweeps workzone parameters (taper length, cone spacing, speed limit) and
# evaluates safety metrics (cone near-miss rate, speed violations, collision
# rate) using the GPU-vectorized SceneFactory RL environment.
#
# Prerequisites
# -------------
# 1. A trained MARL policy checkpoint (e.g., logs/.../model_XXX.pt).
# 2. configs/scene_factory/workzone_train.yaml exists (created by WZ-OPT-04).
# 3. IsaacLab conda environment is activated (or PYTHONPATH is set).
#
# Usage
# -----
#   bash run_workzone_optimize.sh
#   bash run_workzone_optimize.sh --checkpoint <path> --cem_iterations 30
#   bash run_workzone_optimize.sh --population_size 32 --elite_fraction 0.2
#
# All extra arguments after "--" are forwarded verbatim to the optimizer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Defaults (override via environment variables or CLI args below)
# ---------------------------------------------------------------------------
CHECKPOINT="${CHECKPOINT:-logs/rsl_rl/workzone_optimization/workzone_train/model_300.pt}"
CONFIG="${CONFIG:-configs/scene_factory/workzone_train.yaml}"
SCENE_DIR="${SCENE_DIR:-data/processed/workzone_scenes_json}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/workzone_optimize}"
N_SCENES="${N_SCENES:-8}"
CEM_ITERATIONS="${CEM_ITERATIONS:-20}"
POPULATION_SIZE="${POPULATION_SIZE:-16}"
ELITE_FRACTION="${ELITE_FRACTION:-0.25}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"
EVAL_TIMEOUT_S="${EVAL_TIMEOUT_S:-900}"

# ---------------------------------------------------------------------------
# Parse known --key value overrides (simple passthrough approach)
# ---------------------------------------------------------------------------
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)    CHECKPOINT="$2";     shift 2;;
    --config)        CONFIG="$2";         shift 2;;
    --scene_dir)     SCENE_DIR="$2";      shift 2;;
    --output_dir)    OUTPUT_DIR="$2";     shift 2;;
    --n_scenes)      N_SCENES="$2";       shift 2;;
    --cem_iterations) CEM_ITERATIONS="$2"; shift 2;;
    --population_size) POPULATION_SIZE="$2"; shift 2;;
    --elite_fraction) ELITE_FRACTION="$2"; shift 2;;
    --seed)          SEED="$2";           shift 2;;
    --device)        DEVICE="$2";         shift 2;;
    --eval_timeout_s) EVAL_TIMEOUT_S="$2"; shift 2;;
    *)               EXTRA_ARGS+=("$1");  shift;;
  esac
done

echo "[WZ-OPT] Workzone CEM optimizer"
echo "  checkpoint   : $CHECKPOINT"
echo "  config       : $CONFIG"
echo "  scene_dir    : $SCENE_DIR"
echo "  output_dir   : $OUTPUT_DIR"
echo "  n_scenes     : $N_SCENES"
echo "  cem_iters    : $CEM_ITERATIONS"
echo "  population   : $POPULATION_SIZE"
echo "  elite_frac   : $ELITE_FRACTION"
echo "  device       : $DEVICE"
echo "  eval_timeout : ${EVAL_TIMEOUT_S}s"

PYTHONPATH=. python -u scripts/optimize_workzone_config.py \
    --checkpoint    "$CHECKPOINT" \
    --config        "$CONFIG" \
    --scene_dir     "$SCENE_DIR" \
    --output_dir    "$OUTPUT_DIR" \
    --n_scenes      "$N_SCENES" \
    --cem_iterations "$CEM_ITERATIONS" \
    --population_size "$POPULATION_SIZE" \
    --elite_fraction "$ELITE_FRACTION" \
    --seed          "$SEED" \
    --device        "$DEVICE" \
    --eval_timeout_s "$EVAL_TIMEOUT_S" \
    "${EXTRA_ARGS[@]}"
