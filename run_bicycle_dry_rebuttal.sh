#!/usr/bin/env bash
# DRY arm on the KINEMATIC BICYCLE backend, speed-matched to the PhysX vehicle.
#
# Counterpart to the PhysX dry arm trained by run_weather_ablation_rebuttal.sh.
# The two configs differ in exactly four substantive lines (dynamics_mode,
# bicycle_max_speed_mps, run_name, max_iterations); scene pool, observation
# space, policy, and reward are byte-identical. Verify any time with:
#   diff <(grep -v '^\s*#' configs/scene_factory/waymo_physx_256_train_16agents_knn_alldry.yaml) \
#        <(grep -v '^\s*#' configs/scene_factory/waymo_physx_256_train_16agents_knn_alldry_bicycle.yaml)
#
# 900 iterations to match the PhysX dry checkpoint used in the Table 4 rerun
# (model_900), so the two backends are compared at the same training budget.
# Both are well past the ~600 plateau.
#
# bicycle_max_speed_mps=4.5 matches the PhysX vehicle's MEASURED 4.43 m/s rather
# than the shared 15.0 default, which the PhysX backend never reaches because it
# is traction-limited. See the config header for why this matters and for the
# one axis still unmatched (bicycle_accel_scale 6.0 vs ~1.8 m/s^2 measured).
#
# Usage:
#   bash run_bicycle_dry_rebuttal.sh [SEED] [GPU]
#
# Checkpoints: logs/rsl_rl/waymo_physx_256/<stamp>_knn_16agents_alldry_bicycle_s<seed>/
# Console log: artifacts/bicycle_dry_rebuttal/dry_bicycle_s<seed>_<stamp>.log

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SEED="${1:-1}"
GPU="${2:-0}"
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "usage: bash $0 [SEED] [GPU]"; exit 2; }

CONFIG=configs/scene_factory/waymo_physx_256_train_16agents_knn_alldry_bicycle.yaml
POOL=configs/scene_factory/generated/scene_factory_256scene_0414_alldry.yaml
# Interpreter. Defaults to whatever python is active, which is what you want
# after `conda activate scenefactory`. Override with SF_PYTHON if your Isaac
# Sim environment is not the active one.
PY="${SF_PYTHON:-$(command -v python)}"
[ -x "$PY" ] || PY=python
[ -f "$CONFIG" ] || { echo "FATAL: config missing: $CONFIG"; exit 1; }
[ -f "$POOL" ]   || { echo "FATAL: scene pool missing: $POOL"; exit 1; }

RUN_NAME="knn_16agents_alldry_bicycle_s${SEED}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR=artifacts/bicycle_dry_rebuttal
mkdir -p "$LOGDIR"
LOG="$LOGDIR/dry_bicycle_s${SEED}_${STAMP}.log"

{
  echo "=== dry arm, kinematic bicycle backend (speed-matched) ==="
  date
  echo "config   : $CONFIG"
  echo "pool     : $POOL"
  echo "seed     : $SEED   (training seed only; world assignment fixed at 42)"
  echo "run_name : $RUN_NAME"
  echo "gpu      : $GPU"
  echo "iters    : 900   v_max : 4.5 m/s (PhysX measured 4.43)"
  echo "--- substantive diff vs the PhysX dry arm (expect 4 lines) ---"
  diff <(grep -v '^\s*#' configs/scene_factory/waymo_physx_256_train_16agents_knn_alldry.yaml | grep -v '^\s*$') \
       <(grep -v '^\s*#' "$CONFIG" | grep -v '^\s*$') || true
  echo "-------------------------------------------------------------"
} 2>&1 | tee "$LOG"

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PY" -u \
  src/train_student_vehicle_goal_multiagent_rsl_rl.py \
  --config "$CONFIG" \
  --dynamics_mode bicycle \
  --seed "$SEED" \
  --run_name "$RUN_NAME" \
  --max_iterations 900 \
  --headless \
  "${@:3}" 2>&1 | tee -a "$LOG"

echo "exit=${PIPESTATUS[0]}  log=$LOG" | tee -a "$LOG"
