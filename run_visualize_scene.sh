#!/usr/bin/env bash
# ============================================================
#  run_visualize_scene.sh
#  Open Isaac Sim GUI with SceneFactory roads + fitted vehicles
#  for illustration / screenshot purposes.
#
#  Usage:
#    bash run_visualize_scene.sh                   # default 4 worlds, GUI
#    bash run_visualize_scene.sh --world_count 1   # single world
#    bash run_visualize_scene.sh --save_stage_usd  # GUI + export .usda on exit
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/scene_factory/visualize_scene.yaml"
OUTPUT_DIR="artifacts/scene_factory/visualize_scene"

# ---- locate python / isaaclab python -----------------------
if command -v isaaclab &>/dev/null; then
    PYTHON_CMD="isaaclab -p"
elif [ -f "${ISAACLAB_PATH:-}/isaaclab.sh" ]; then
    PYTHON_CMD="${ISAACLAB_PATH}/isaaclab.sh -p"
else
    PYTHON_CMD="python"
fi

echo "=========================================="
echo " SceneFactory — GUI Visualizer"
echo " Config : $CONFIG"
echo " Output : $OUTPUT_DIR"
echo "=========================================="

$PYTHON_CMD -m src.scene_factory_multiworld_scene \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    "$@"
