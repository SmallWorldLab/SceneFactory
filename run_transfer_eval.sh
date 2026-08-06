#!/usr/bin/env bash
# Cross-dynamics transfer evaluation (rebuttal; bm7r Q5 / B4, paper Table 3).
#
# 2 policies x 2 evaluation backends x N seeds, all on DRY ground.
#
#                         eval backend
#                    physx            bicycle
#   physx policy   NATIVE           TRANSFER  physx -> bicycle
#   bicycle policy TRANSFER         NATIVE
#                  bicycle -> physx
#
# The two native cells are the controls: they establish each policy's own-backend
# performance so a transfer drop is measured against the right reference rather
# than against the other policy.
#
# SPEED PARAMETER -- READ BEFORE CHANGING
# ---------------------------------------
# bicycle_max_speed_mps means different things per backend:
#   physx   : wheel VELOCITY TARGET. Body is traction-limited to ~4.43 m/s
#             (measured, artifacts/traction_probe/rad_cub1000) at the 15.0 value
#             both PhysX policies trained with.
#   bicycle : hard, reachable clamp on body speed.
# So the value is set PER BACKEND to whatever that backend was trained with:
# 15.0 for physx, 4.5 for bicycle. Matching the NUMBERS would un-match the
# VEHICLES. Using 15.0 under the bicycle backend would give a ~3.4x faster car --
# the confound present in the submitted Table 3, whose bicycle config also used
# the default.
#
# NOTE: until 2026-07-27 no CLI flag existed for this field and it was honoured
# only inside test_mode branches, so bicycle_max_speed_mps in a training or eval
# YAML was silently ignored. --bicycle_max_speed_mps now applies it everywhere.
#
# Test set: 64 worlds drawn per seed from the 199-scene held-out pool (disjoint
# from the 256 training scenes). Seeds vary the map draw, so spread across seeds
# is unseen-map variance. Both policies see the SAME draw at each seed, so cells
# can be compared paired.
#
# Train/eval parity is forced on the command line for the same reasons as
# run_paper_table4_eval.sh (the shared eval config was written for the E
# backbone): wheel_friction_cap 1.2, ground_cuboid_size_m 1000, env_spacing 1300.
# Under the bicycle backend those three are inert (no contact) but harmless.
#
# Weather token is blinded throughout: both policies are dry-trained and have
# never seen a nonzero water film, so a live token would be an out-of-distribution
# input. On the dry pool the blinded constant [0,1,0,0] is bit-identical to the
# live token anyway.
#
# Usage:
#   bash run_transfer_eval.sh [NUM_SEEDS] [GPU] [invincible|terminating]
#
# Aggregate: PYTHONPATH=. python scripts/aggregate_transfer_eval.py

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NUM_SEEDS="${1:-5}"
GPU="${2:-0}"
PROTOCOL="${3:-invincible}"
case "$PROTOCOL" in
  invincible)  PROT_FLAG="--invincible" ;;
  terminating) PROT_FLAG="--no-invincible" ;;
  *) echo "usage: bash $0 [NUM_SEEDS] [GPU] [invincible|terminating]"; exit 2 ;;
esac

# Interpreter. Defaults to whatever python is active, which is what you want
# after `conda activate scenefactory`. Override with SF_PYTHON if your Isaac
# Sim environment is not the active one.
PY="${SF_PYTHON:-$(command -v python)}"
[ -x "$PY" ] || PY=python
CFG=configs/scene_factory/eval_knn_backbone_weather_multiseed.yaml
POOL=configs/scene_factory/generated/eval_unseen_199scenes_dry.yaml
LOGDIR=artifacts/transfer_eval
mkdir -p "$LOGDIR"

last_ckpt () {  # $1 = run dir -> highest-numbered model_*.pt
  ls -1 "$1"/model_*.pt 2>/dev/null \
    | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-
}

# PhysX arm: the shipped policy. Override with CKPT_PHYSX to use your own run.
CKPT_PHYSX="${CKPT_PHYSX:-checkpoints/dry_trained_iter900.pt}"

# Bicycle arm: not shipped -- train it with run_bicycle_dry_rebuttal.sh, which
# takes ~1 GPU-day. Must be SPEED-MATCHED to the PhysX vehicle (v_max 4.5); an
# unmatched bicycle policy drives a different problem and the transfer number
# means nothing. Match on the recorded config rather than on mtime, and take the
# last checkpoint rather than a hardcoded iteration -- the arms stop at different
# iterations and hardcoding one silently fails on the other.
if [ -z "${CKPT_BICYCLE:-}" ]; then
  DIR_BICYCLE=""
  for d in logs/rsl_rl/waymo_physx_256/*_knn_16agents_alldry_bicycle*_s1; do
    [ -d "$d" ] || continue
    grep -q "bicycle_max_speed_mps: 4.5" "$d/params/env.yaml" 2>/dev/null && DIR_BICYCLE="$d"
  done
  [ -n "$DIR_BICYCLE" ] || { echo "FATAL: no bicycle run with bicycle_max_speed_mps=4.5 found."; \
    echo "       train it:  bash run_bicycle_dry_rebuttal.sh 1 0"; \
    echo "       or point at your own:  CKPT_BICYCLE=path/to/model.pt bash $0"; exit 1; }
  CKPT_BICYCLE=$(last_ckpt "$DIR_BICYCLE")
fi

[ -f "$CFG" ]  || { echo "FATAL: eval config missing: $CFG"; exit 1; }
[ -f "$POOL" ] || { echo "FATAL: dry eval pool missing: $POOL"; exit 1; }
[ -f "$CKPT_PHYSX" ]   || { echo "FATAL: physx checkpoint missing: $CKPT_PHYSX"; exit 1; }
[ -n "$CKPT_BICYCLE" ] && [ -f "$CKPT_BICYCLE" ] || { echo "FATAL: bicycle checkpoint missing: ${CKPT_BICYCLE:-<unset>}"; exit 1; }

echo "=== transfer eval: 2 policies x 2 backends x ${NUM_SEEDS} seeds, ${PROTOCOL}, GPU ${GPU} ==="
echo "physx   policy: $CKPT_PHYSX"
echo "bicycle policy: $CKPT_BICYCLE"
date

for policy in physx bicycle; do
  case "$policy" in
    physx)   CKPT="$CKPT_PHYSX" ;;
    bicycle) CKPT="$CKPT_BICYCLE" ;;
  esac
  for backend in physx bicycle; do
    # Per-backend speed parameter: whatever that backend was trained with.
    case "$backend" in
      physx)   VMAX=15.0 ;;
      bicycle) VMAX=4.5  ;;
    esac
    for s in $(seq 1 "$NUM_SEEDS"); do
      seed=$((100 + s))
      tag="tx_${policy}2${backend}_seed${seed}"
      done_dir=""
      for c in logs/rsl_rl/transfer_eval/*_"${tag}"; do
        [ -d "$c" ] && [ -f "$c/outcome.json" ] && \
          grep -q '"status": "success"' "$c/outcome.json" 2>/dev/null && done_dir="$c"
      done
      if [ -n "$done_dir" ]; then
        echo "--- ${tag}  SKIP (already collected)"; continue
      fi
      echo "--- ${tag}  (backend=${backend}, v_max=${VMAX}) ---"
      start=$(date +%s)
      CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PY" -u \
        src/train_student_vehicle_goal_multiagent_rsl_rl.py \
        --config "$CFG" \
        --scene_factory_config "$POOL" \
        --test_mode scene_factory_policy_eval \
        --checkpoint_path "$CKPT" \
        --dynamics_mode "$backend" \
        --bicycle_max_speed_mps "$VMAX" \
        --headless --no-use_fabric \
        $PROT_FLAG --no-random_od \
        --obs_weather_context_blind \
        --wheel_friction_cap 1.2 \
        --ground_cuboid_size_m 1000 \
        --env_spacing 1300 \
        --seed "$seed" \
        --scene_factory_random_world_seed "$seed" \
        --experiment_name "transfer_eval" \
        --run_name "$tag" \
        > "${LOGDIR}/${tag}.log" 2>&1
      echo "    exit=$?  $(( $(date +%s) - start ))s"
    done
  done
done

echo "=== done ==="; date
echo "Aggregate: PYTHONPATH=. python scripts/aggregate_transfer_eval.py"
