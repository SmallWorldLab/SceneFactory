#!/usr/bin/env bash
# run_friction_sweep_eval.sh
# ─────────────────────────────────────────────────────────────────────────────
# Friction ablation: same 64 held-out Waymo scenes, 4 friction conditions,
# evaluated in two modes:
#   conditioned — policy sees real friction token in observation
#   blind       — friction token zeroed (policy thinks it's always dry AC)
#
# ground_mode=cuboid in all configs so friction is physically active.
#
# Friction conditions (AC surface):
#   dry           AC  0 mm   mu=1.10   full grip
#   light_rain    AC  0.3mm  mu=0.96   slight reduction
#   moderate_rain AC  0.6mm  mu=0.83   meaningful reduction
#   heavy_rain    AC  1.0mm  mu=0.00   full hydroplaning
#
# Checkpoint: friction_16agents model_2999 (weather-conditioned policy)
#
# Usage:
#   bash run_friction_sweep_eval.sh                        # all 8 runs
#   bash run_friction_sweep_eval.sh --cond dry             # one condition, both modes
#   bash run_friction_sweep_eval.sh --cond dry --mode conditioned  # single run
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

CHECKPOINT="logs/rsl_rl/waymo_physx_256/2026-06-04_10-32-07_friction_16agents/model_2999.pt"
PYTHON="src/train_student_vehicle_goal_multiagent_rsl_rl.py"
COMMON="--headless --no-use_fabric --device cuda:0 --test_mode scene_factory_policy_eval --checkpoint_path $CHECKPOINT"

COND="ALL"
MODE="ALL"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cond) COND="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

run_eval() {
    local tag="$1"
    local label="$2"
    local mu="$3"
    local blind="$4"   # "conditioned" or "blind"
    local suffix=""
    [[ "$blind" == "blind" ]] && suffix="_blind"
    local cfg="configs/scene_factory/generated/eval_friction_sweep_${tag}${suffix}.yaml"
    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  ${label}  (${mu})  [${blind}]"
    echo "══════════════════════════════════════════════════"
    PYTHONPATH=. python -u $PYTHON $COMMON --config "$cfg"
}

CONDITIONS=(
    "dry           AC dry 0mm      mu=1.10"
    "light_rain    AC rain 0.3mm   mu=0.96"
    "moderate_rain AC rain 0.6mm   mu=0.83"
    "heavy_rain    AC rain 1.0mm   mu=0.00"
)

for entry in "${CONDITIONS[@]}"; do
    read -r tag label mu <<< "$entry"
    [[ "$COND" != "ALL" && "$COND" != "$tag" ]] && continue
    [[ "$MODE" == "ALL" || "$MODE" == "conditioned" ]] && run_eval "$tag" "$label" "$mu" "conditioned"
    [[ "$MODE" == "ALL" || "$MODE" == "blind"       ]] && run_eval "$tag" "$label" "$mu" "blind"
done

# ── Summary table ─────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Results summary"
echo "  Condition            │ Conditioned SR │ Blind SR │ Δ SR"
echo "  ─────────────────────┼────────────────┼──────────┼──────"

get_sr() {
    local pattern="$1"
    local latest
    latest=$(ls -td logs/rsl_rl/waymo_physx_256_eval/*${pattern}* 2>/dev/null | head -1)
    if [[ -n "$latest" && -f "$latest/scene_factory_policy_eval_summary.json" ]]; then
        python3 -c "import json; d=json.load(open('$latest/scene_factory_policy_eval_summary.json')); print(f'{d[\"success_rate\"]:.1%}')" 2>/dev/null
    else
        echo "  n/a "
    fi
}

for entry in "${CONDITIONS[@]}"; do
    read -r tag label mu <<< "$entry"
    sr_cond=$(get_sr "friction_sweep_${tag}_model2999")
    sr_blind=$(get_sr "friction_sweep_${tag}_blind_model2999")
    printf "  %-20s │ %-14s │ %-8s │\n" "$label ($mu)" "$sr_cond" "$sr_blind"
done
echo "════════════════════════════════════════════════════════════════════"
