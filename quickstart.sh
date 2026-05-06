#!/usr/bin/env bash
# SceneFactory quickstart — verifies setup and runs a 4-world visualization
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  SceneFactory Quickstart"
echo "=========================================="
echo ""

# 1. Check isaaclab.sh is available
if ! command -v isaaclab.sh &>/dev/null; then
  echo "ERROR: 'isaaclab.sh' not found on PATH."
  echo "Please install Isaac Lab and add it to your PATH."
  echo "  https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html"
  exit 1
fi
echo "[1/3] isaaclab.sh found: $(which isaaclab.sh)"

# 2. Check scene data exists
SCENE_DIR="data/processed/waymo_scenes_json"
if [[ ! -d "$SCENE_DIR" ]] || [[ -z "$(ls -A "$SCENE_DIR" 2>/dev/null)" ]]; then
  echo ""
  echo "WARNING: Scene data not found at '$SCENE_DIR'."
  echo "Run the Waymo preprocessing pipeline first:"
  echo ""
  echo "  isaaclab.sh -p -m src.trfc.world_pipeline \\"
  echo "    --tfrecord-dir /path/to/waymo_tfrecords \\"
  echo "    --output-dir $SCENE_DIR"
  echo ""
  exit 1
fi
NSCENES=$(ls "$SCENE_DIR"/*.json 2>/dev/null | wc -l)
echo "[2/3] Scene data found: $NSCENES JSON scenes in $SCENE_DIR"

# 3. Launch 4-world visualization
echo "[3/3] Launching 4-world visualization..."
echo ""
PYTHONPATH=. isaaclab.sh -p src/run_student_vehicle_goal_multiagent_random.py \
  --config configs/scene_factory/visualize_scene.yaml \
  --world_count 4 \
  "$@"
