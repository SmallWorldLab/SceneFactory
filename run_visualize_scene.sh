#!/usr/bin/env bash
# ============================================================
#  run_visualize_scene.sh
#  Open Isaac Sim GUI with SceneFactory roads + fitted vehicles
#  for illustration / screenshot purposes.
#
#  Usage:
#    bash run_visualize_scene.sh                   # default 4 worlds, GUI
#    bash run_visualize_scene.sh --world_count 64  # 8×8 grid of unique scenes
#    bash run_visualize_scene.sh --world_count 1 --freeze  # freeze after 10 steps (good for screenshots)
#    bash run_visualize_scene.sh --save_stage_usd  # GUI + export .usda on exit
#
#  Off-screen camera render (no GUI window):
#    bash run_visualize_scene.sh --headless \
#      --camera_path artifacts/camera_traj.json \
#      --capture_dir artifacts/render_out \
#      --capture_fps 30 --capture_width 1920 --capture_height 1080
#
#  Camera trajectory JSON format (artifacts/camera_traj.json):
#    [
#      {"t": 0.0,  "eye": [200, 200, 160], "lookat": [0, 0, 0]},
#      {"t": 5.0,  "eye": [100,  50,  80], "lookat": [0, 0, 0]},
#      {"t": 10.0, "eye": [  0, 200,  40], "lookat": [0, 0, 0]}
#    ]
#    t = time in seconds, eye/lookat in world-space metres.
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
