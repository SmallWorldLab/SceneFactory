#!/usr/bin/env bash
# Run the brake friction sweep N times with different seeds, to show the per-mu
# spread (rule out a single-seed / solver-noise fluke). Mirrors NHTSA's replications.
#
# Usage:
#   bash run_brake_sweep_replicates.sh [N=10]
# Env:
#   BFS_V0_MPS (default 27.78 = 100 km/h), BFS_MAX_BRAKE_STEPS (2000)
#
# Each replicate is an independent full sweep (fresh Isaac boot, seed=1..N). The
# per-seed summary CSVs are collected into one replicates dir for aggregation by
#   python scripts/plot_brake_sweep.py --replicates <dir>

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

N="${1:-10}"
V0="${BFS_V0_MPS:-27.78}"
MAXB="${BFS_MAX_BRAKE_STEPS:-2000}"
TS=$(date +%Y%m%d_%H%M%S)
REPDIR="logs/rsl_rl/brake_friction_sweep/replicates_${TS}"
mkdir -p "$REPDIR"
echo "[replicates] N=$N  v0=$V0  -> $REPDIR"

ok=0
for s in $(seq 1 "$N"); do
    echo "[replicates] seed=$s starting ..."
    CUDA_VISIBLE_DEVICES=0 BFS_V0_MPS="$V0" BFS_MAX_BRAKE_STEPS="$MAXB" \
        conda run -n isaac-pytorch --no-capture-output \
        bash run_brake_friction_sweep.sh --seed "$s" > "$REPDIR/run_seed${s}.log" 2>&1
    rc=$?
    RUN=$(ls -td logs/rsl_rl/brake_friction_sweep/*_physx_mu_sweep 2>/dev/null | head -1)
    if [ $rc -eq 0 ] && [ -f "$RUN/brake_sweep_summary.csv" ]; then
        cp "$RUN/brake_sweep_summary.csv" "$REPDIR/summary_seed${s}.csv"
        ok=$((ok+1))
        echo "[replicates] seed=$s done -> $(basename "$RUN")"
    else
        echo "[replicates] seed=$s FAILED (rc=$rc) -- see $REPDIR/run_seed${s}.log"
    fi
done

echo "[replicates] collected $ok/$N summaries in $REPDIR"
echo "[replicates] plot with: python scripts/plot_brake_sweep.py --replicates $REPDIR"
