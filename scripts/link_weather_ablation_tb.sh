#!/usr/bin/env bash
# Refresh symlinks so TensorBoard shows only the CURRENT rebuttal ablation runs.
# Re-run whenever a new seed starts -- TensorBoard picks it up live, no restart.
# Runs containing a .aborted marker file are skipped.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TB=artifacts/weather_ablation_rebuttal/tb
mkdir -p "$TB"; find "$TB" -type l -delete
for d in logs/rsl_rl/waymo_physx_256/*_knn_16agents_{alldry,wet0to12}_s[0-9]*; do
  [ -d "$d" ] || continue
  if [ -f "$d/.aborted" ]; then echo "skip (aborted): $(basename "$d")"; continue; fi
  b=$(basename "$d"); stamp=${b:5:11}; stamp=${stamp//-/}; arm=${b#*_knn_16agents_}
  ln -sfn "$(cd "$d" && pwd)" "$TB/${arm}__${stamp}"
  echo "linked ${arm}__${stamp}"
done
