#!/usr/bin/env bash
# Traction probe: 2x2 isolation of why the weather-ablation runs do not drive.
#
# The ablation configs differ from the E backbone's in exactly two keys.  This
# grid varies ONLY those two, on ONE scene pool (alldry, mu ~1.18 uniform, no
# frictionless worlds), so the pool cannot confound the result.
#
#   variant       ground_cuboid_size_m   wheel_friction_cap   =
#   e_baseline           1000                   1.0             E's effective config
#   cuboid_only           300                   1.0             cuboid fix alone
#   cap_only             1000                   1.2             cap alone
#   both                  300                   1.2             the new training config
#
# cap=1.0 is a documented no-op (it equals the value baked into the vehicle USD).
# cuboid=1000 with env_spacing=400 is the OVERLAPPING configuration -- which is
# what E trained on.
#
# Read the VERDICT line and the per-agent-index table:
#   wheels spin + body still  -> zero normal force, ground contact defect
#   wheels still + body still -> drive command never reached the joints
#   agent 0 drives, 1..15 not -> the documented multi-agent penetration bug
#
# Usage:
#   bash run_traction_probe.sh [variant] [gpu]      # one variant
#   bash run_traction_probe.sh all [gpu]            # all four, sequentially
#
# Results: artifacts/traction_probe/<variant>/traction_probe.{csv,json}

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VARIANT="${1:-all}"
GPU="${2:-1}"
CONFIG=configs/scene_factory/waymo_physx_256_train_16agents_knn_alldry.yaml
PY=/home/yz8733/miniforge3/envs/isaac-pytorch/bin/python
[ -x "$PY" ] || PY=python
[ -f "$CONFIG" ] || { echo "FATAL: config missing: $CONFIG"; exit 1; }

NUM_ENVS="${TP_NUM_ENVS:-8}"      # 8 envs x 16 agents = 128 agents, plenty of samples

run_one () {
  local name="$1" cuboid="$2" cap="$3"
  local out="artifacts/traction_probe/${name}"
  mkdir -p "$out"
  echo
  echo "================================================================"
  echo "  variant=${name}  ground_cuboid_size_m=${cuboid}  wheel_friction_cap=${cap}"
  echo "================================================================"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PY" -u \
    src/train_student_vehicle_goal_multiagent_rsl_rl.py \
    --config "$CONFIG" \
    --test_mode traction_probe \
    --num_envs "$NUM_ENVS" \
    --ground_cuboid_size_m "$cuboid" \
    --wheel_friction_cap "$cap" \
    --log_dir "$out" \
    --headless 2>&1 | tee "${out}/probe.log"
  # The probe writes into the run_dir the trainer creates under --log_dir.
  find "$out" -name traction_probe.json -newermt '-10 minutes' -exec sh -c \
    'echo "--- $1 ---"; cat "$1"' _ {} \; 2>/dev/null | tail -30
}

case "$VARIANT" in
  e_baseline)  run_one e_baseline  1000 1.0 ;;
  cuboid_only) run_one cuboid_only  300 1.0 ;;
  cap_only)    run_one cap_only    1000 1.2 ;;
  both)        run_one both         300 1.2 ;;
  all)
    run_one e_baseline  1000 1.0
    run_one cuboid_only  300 1.0
    run_one cap_only    1000 1.2
    run_one both         300 1.2
    echo
    echo "=== GRID SUMMARY ==="
    for v in e_baseline cuboid_only cap_only both; do
      f=$(find "artifacts/traction_probe/${v}" -name traction_probe.json 2>/dev/null | head -1)
      if [ -n "$f" ]; then
        "$PY" - "$v" "$f" <<'EOF'
import json,sys
v,f=sys.argv[1],sys.argv[2]; d=json.load(open(f))
print(f"{v:<12} speed {d['mean_speed_mps']:>6.3f} m/s  omega {d['mean_wheel_omega_radps']:>6.2f}  "
      f"slip {d['mean_slip']:>5.3f}  idle {100*d['idle_fraction_overall']:>5.1f}%  "
      f"(a0 {100*d['idle_fraction_agent0']:>5.1f}% / a1+ {100*d['idle_fraction_agents_1plus']:>5.1f}%)")
EOF
      else
        echo "$v: NO RESULT"
      fi
    done
    ;;
  *) echo "usage: bash $0 [e_baseline|cuboid_only|cap_only|both|all] [gpu]"; exit 2 ;;
esac
