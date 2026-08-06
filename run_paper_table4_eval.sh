#!/usr/bin/env bash
# Reproduce the paper's Table 4 (§4.5 friction conditioning) with the RETRAINED
# policies and the CORRECTED friction model.
#
# Policies under test -- the two arms of the weather-exposure ablation, taken at
# the SAME iteration (900) because dry_s1 was killed by a CUDA launch timeout at
# ~900 while wet_s1 reached 1100. Both are past plateau (~600).
#
#   blind  alldry_s1/model_900.pt     trained on mu ~1.18 everywhere (v8 analogue)
#   aware  wet0to12_s1/model_900.pt   trained on mu 0.41-1.20        (v7 analogue)
#
# Both configs have weather_context_enable: true, so they differ in TRAINING
# EXPOSURE, exactly as v7/v8 did. The weather token is then set PER POLICY at eval
# time -- blinded for the dry-only policy, live for the wet-exposed one. See the
# rationale in the policy loop below; this is not a free axis, it is what makes
# the dry-only baseline fair rather than fed an input it never trained on.
#
# Test set: the held-out 199-scene pool, 64 drawn per run via
# --scene_factory_random_world_seed, so across-seed spread is map-sampling
# variance -- the "unseen maps" quantity the paper reports and the multi-seed
# uncertainty reviewers asked for (AC priority #3, bm7r Q2).
#
# Surfaces (corrected TRFC, mu at the 13.89 m/s reference speed):
#   dry       AC  0.0 mm   mu 1.174     (Table 4 "Dry")
#   wet0p5mm  AC  0.5 mm   mu 0.944     (Table 4 "Moderate wet")
#   wet5mm    AC  5.0 mm   mu 0.506     NEW -- below the ~0.78 braking threshold
#   wet12mm   AC 12.0 mm   mu 0.408     NEW -- deep into the binding region
#
# The last two exist because the published wet cell (mu 0.86-0.94) never reached
# the regime where friction binds on this vehicle. If conditioning has an effect
# at all it must appear at 5-12 mm.
#
# PROTOCOL defaults to `invincible` to match Table 4 ("invincible mode"). Pass
# `terminating` for the stricter protocol, where a crash ends the episode --
# expect SR ~20 points lower and report both if there is time.
#
# TRAIN/EVAL PARITY (audited 2026-07-26 before launch).  The shared eval config
# eval_knn_backbone_weather_multiseed.yaml was written for the E backbone, not for
# these policies, so the three physics keys below are forced on the command line
# rather than trusted from that file:
#
#   --wheel_friction_cap 1.2    eval config omits it -> defaults to 1.0.  Contact
#                               mu = min(ground_mu, cap), so the DRY cell would
#                               have run at mu 1.0 while it trained at 1.174.
#                               Wet cells (mu <= 0.944) were unaffected, i.e. this
#                               would have biased only the baseline row.
#   --ground_cuboid_size_m 1000 same as the eval default; pinned to be explicit.
#   --env_spacing 1300          eval config says 1500.  Physically equivalent
#                               (both > slab, so non-overlapping) but matched anyway.
#
# Verified IDENTICAL and needing no override: every observation key (road_points_k
# 250, weather_context_enable, road_points_include_dirs), the policy block,
# dynamics_mode (physx), observation_mode, reset_mode, spawn_height_m,
# max_distance_from_origin_m, goal_reached_threshold_m, num_agents_per_env.
# ground_mode differs on paper ("plane" vs "cuboid") but is a DEAD FIELD --
# _spawn_ground(..., mode="cuboid") is hardcoded at multiagent env :1548.
#
# DELIBERATELY NOT matched: episode_length_s is 45 in training and 50 here. That
# is the paper's eval protocol; changing it would break comparability with the
# published numbers. Eval simply allows 5 s more than training.
#
# TEST SET IS UNSEEN: the 256 training scenes and the 199 eval scenes have ZERO
# overlap (verified by scene_json set intersection). All four eval weather pools
# carry identical scene sets and differ only in water_film_mm.
#
# Usage:
#   bash run_paper_table4_eval.sh [NUM_SEEDS] [GPU] [invincible|terminating]
#
# Aggregate:  PYTHONPATH=. python scripts/aggregate_table4_eval.py

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
RUNROOT=logs/rsl_rl/table4_eval
LOGDIR=artifacts/table4_eval
mkdir -p "$LOGDIR"

# The two shipped policies. Override to evaluate your own training runs:
#   CKPT_BLIND=logs/rsl_rl/.../model_900.pt bash run_paper_table4_eval.sh
CKPT_BLIND="${CKPT_BLIND:-checkpoints/dry_trained_iter900.pt}"
CKPT_AWARE="${CKPT_AWARE:-checkpoints/weather_exposed_iter900.pt}"

[ -f "$CFG" ]        || { echo "FATAL: config missing: $CFG"; exit 1; }
[ -f "$CKPT_BLIND" ] || { echo "FATAL: checkpoint missing: $CKPT_BLIND"; exit 1; }
[ -f "$CKPT_AWARE" ] || { echo "FATAL: checkpoint missing: $CKPT_AWARE"; exit 1; }

echo "=== Table 4 eval: 2 policies x 4 surfaces x ${NUM_SEEDS} seeds, ${PROTOCOL}, GPU ${GPU} ==="
date

for policy in blind aware; do
  # WEATHER-TOKEN POLICY (fixed 2026-07-26 after a contaminated first sweep).
  #
  # The dry-only policy trained on the alldry pool, where water_film_mm = 0.0 in
  # EVERY world, so it has never seen a nonzero first element of the weather
  # token. Feeding it the live token on a wet world is an OUT-OF-DISTRIBUTION
  # INPUT, not a harder road, and it collapsed SR 94.2 -> 35.4 at mu 0.944 --
  # a regime where friction provably does not bind (brake sweeps show no
  # separation until mu < 0.5). That was an artifact of the ablation design.
  #
  # So the dry-only policy is evaluated with its weather channel BLINDED in every
  # condition, which is the fair baseline: it drives the wet road without being
  # handed an input it cannot interpret. Blinding pins the token to the dry-AC
  # constant [0,1,0,0], which is bit-identical to the live token on the dry pool
  # (verified) -- so the already-completed blind_dry cells remain valid.
  #
  # The wet-exposed policy saw water films 0-12 mm in training, so every token
  # value is in distribution. It keeps the live token: using weather information
  # is exactly the capability under test.
  case "$policy" in
    blind) CKPT="$CKPT_BLIND"; TOKEN_FLAG="--obs_weather_context_blind" ;;
    aware) CKPT="$CKPT_AWARE"; TOKEN_FLAG="--no-obs_weather_context_blind" ;;
  esac
  for wx in dry wet0p5mm wet5mm wet12mm; do
    POOL="configs/scene_factory/generated/eval_unseen_199scenes_${wx}.yaml"
    [ -f "$POOL" ] || { echo "FATAL: pool missing: $POOL (run scripts/make_eval_weather_pools.py)"; exit 1; }
    for s in $(seq 1 "$NUM_SEEDS"); do
      seed=$((100 + s))
      tag="t4_${policy}_${wx}_seed${seed}"
      # RESUME: skip any cell that already has a finished, successful run.
      # Quarantined dirs are renamed (*_INVALID_*, *_PREFIX_*) so they never
      # match here and are always recollected.
      done_dir=""
      for c in logs/rsl_rl/table4_eval/*_"${tag}"; do
        [ -d "$c" ] || continue
        [ -f "$c/outcome.json" ] || continue
        grep -q '"status": "success"' "$c/outcome.json" 2>/dev/null && done_dir="$c"
      done
      if [ -n "$done_dir" ]; then
        echo "--- ${tag}  SKIP (already collected: $(basename "$done_dir"))"
        continue
      fi
      echo "--- ${tag} ---"
      start=$(date +%s)
      CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PY" -u \
        src/train_student_vehicle_goal_multiagent_rsl_rl.py \
        --config "$CFG" \
        --scene_factory_config "$POOL" \
        --test_mode scene_factory_policy_eval \
        --checkpoint_path "$CKPT" \
        --headless --no-use_fabric \
        $PROT_FLAG --no-random_od \
        $TOKEN_FLAG \
        --wheel_friction_cap 1.2 \
        --ground_cuboid_size_m 1000 \
        --env_spacing 1300 \
        --seed "$seed" \
        --scene_factory_random_world_seed "$seed" \
        --experiment_name "table4_eval" \
        --run_name "$tag" \
        > "${LOGDIR}/${tag}.log" 2>&1
      rc=$?
      echo "    exit=${rc}  $(( $(date +%s) - start ))s"
    done
  done
done

echo "=== done ==="; date
echo "Aggregate with: PYTHONPATH=. python scripts/aggregate_table4_eval.py"
