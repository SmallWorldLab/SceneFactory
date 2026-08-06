#!/usr/bin/env bash
# Physics sanity test — pure straight-line full-throttle run, no Waymo data required.
#
# Design choices:
#   steering=0.0                      pure straight-line: isolates longitudinal dynamics
#   agent_spawn_circle_radius_m 8.0   spread agents to avoid spawn interpenetration explosion
#   settle_steps 48                   zero-throttle warmup for suspension to settle (~1s)
#   drive_steps 300                   full-throttle run (~6s), matching teacher rollout length
#   max_distance_from_origin_m 2000   prevents boundary-exceeded resets during test
#   goal_radius 5000-6000m            goal is unreachably far — no goal-reached resets
#
# Expected healthy output — teacher at full throttle reaches 7.6 m/s in 6s (still
# accelerating), so the student should show a similar rising speed curve, not a plateau:
#   max_planar_speed_mps  : 5.0–10.0 m/s  (after 300 drive steps ≈ 6s)
#   mean_planar_speed_mps : 2.0–6.0 m/s
#   root_z_range_m        : < 0.8 m     (grounded, not bouncing)
#
# Diagnosis guide:
#   max < 1.0 m/s  → drive torque not reaching wheels (action pipeline bug)
#   speed plateaus early (< 3 m/s) → viscous friction or contact friction bug
#   speed matches teacher curve (see teacher_speed_curve below) → physics correct
#   z_range > 1.5m → spawn interpenetration or suspension explosion
#
# Teacher reference (straight_accel_t100pct rollout, dt=0.0167s):
#   t=1.5s (step~90):  0.8 m/s
#   t=2.0s (step~120): 1.6 m/s
#   t=3.0s (step~180): 3.1 m/s
#   t=4.0s (step~240): 4.6 m/s
#   t=5.0s (step~300): 6.2 m/s
#   t=6.0s (step~360): 7.6 m/s  (still accelerating)
#
# What this does NOT prove:
#   - Road surface friction  (vehicles drive on ground cuboid only)
#   - Road-edge collision    (enable_segment_collision=False)
#
# IMPORTANT: runs with num_agents_per_env=1 (single agent per world).
# With multiple agents, all spawn facing inward (yaw = formation_angle + pi),
# so they collide with each other at ~t=5s and speed crashes to zero — this
# masks the real physics. Single-agent mode isolates longitudinal dynamics.
#
# Output: logs/rsl_rl/scene_factory_demo/<timestamp>/
#   params/run.json                                  (config + git commit)
#   scene_factory_multiworld_random_steer_test_summary.json   <-- READ THIS
#   scene_factory_multiworld_random_steer_test_metrics.jsonl
#   outcome.json
#
# Usage:
#   bash run_physics_sanity_test.sh                                     # default (plane ground)
#   bash run_physics_sanity_test.sh --config configs/scene_factory/braking_validation.yaml  # cuboid ground

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default config — can be overridden with --config <path>
CONFIG="configs/scene_factory/demo_weather_physx_train.yaml"

# Parse --config argument if provided, pass remaining args through
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            PASSTHROUGH+=("$1")
            shift
            ;;
    esac
done

echo "[physics-sanity] Using config: $CONFIG"

PYTHONPATH=. python -u src/train_student_vehicle_goal_multiagent_rsl_rl.py \
    --config "$CONFIG" \
    --headless \
    --num_envs 4 \
    --num_agents_per_env 1 \
    --test_mode scene_factory_multiworld_random_steer_test \
    --no-use_scene_factory_roads \
    --agent_spawn_circle_radius_m 8.0 \
    --random_steer_test_steering_min 0.0 \
    --random_steer_test_steering_max 0.0 \
    --random_steer_test_settle_steps 48 \
    --random_steer_test_drive_steps 300 \
    --max_distance_from_origin_m 2000.0 \
    --goal_radius_min_m 5000.0 \
    --goal_radius_max_m 6000.0 \
    "${PASSTHROUGH[@]}"

# --- Post-run: print per-step speed curve from the metrics jsonl ---
METRICS=$(find logs/rsl_rl/scene_factory_demo -name "*random_steer_test_metrics.jsonl" \
    -newer run_physics_sanity_test.sh 2>/dev/null | sort | tail -1)
if [ -n "$METRICS" ]; then
    echo ""
    echo "=== Speed curve (env_0 agent_0, drive phase only) ==="
    python3 - "$METRICS" <<'PYEOF'
import json, sys, math

settle = 48
path = sys.argv[1]
print(f"{'step':>6}  {'time(s)':>7}  {'speed(m/s)':>10}  {'teacher_ref':>11}")
print(f"{'------':>6}  {'-------':>7}  {'----------':>10}  {'-----------':>11}")
# Teacher reference speeds at given drive steps (dt=0.0167s per step)
teacher = {0: 0.0, 30: 0.8, 60: 1.6, 120: 3.1, 180: 4.6, 240: 6.2, 300: 7.6}
with open(path) as f:
    for line in f:
        rec = json.loads(line)
        step = rec["step"]
        if step < settle:
            continue
        drive_step = step - settle
        if drive_step % 30 != 0:
            continue
        agents = rec["envs"][0]["agents"]
        agent_id = next(iter(agents))
        spd = agents[agent_id]["planar_speed_mps"]
        t = drive_step * 0.0167
        ref = teacher.get(drive_step, "")
        ref_str = f"{ref:.1f} m/s" if ref != "" else ""
        print(f"{drive_step:>6}  {t:>7.2f}s  {spd:>10.3f}  {ref_str:>11}")
PYEOF
fi
