#!/usr/bin/env bash
# run_workzone_demo.sh
# Launch the workzone taper-zone visualization demo.
#
# Usage:
#   bash run_workzone_demo.sh                    # default config, GUI viewer
#   bash run_workzone_demo.sh --num_worlds 9     # override world count
#   bash run_workzone_demo.sh --headless         # headless (no GUI)
#   bash run_workzone_demo.sh --save_stage_usd   # export stage to artifacts/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_SCRIPT="src/workzone_demo_scene.py"
CONFIG="configs/scene_factory/workzone_demo.yaml"
OUTPUT_DIR="artifacts/scene_factory/workzone_demo"

# Pass all extra CLI args through to the Python script
conda run -n isaac-pytorch --no-capture-output \
    python "$PYTHON_SCRIPT" \
        --config "$CONFIG" \
        --output_dir "$OUTPUT_DIR" \
        "$@"
