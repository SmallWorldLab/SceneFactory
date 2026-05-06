#!/bin/bash
# Quick start script for cross-simulator policy transfer benchmark
# Automatically uses conda environments as needed

set -e

NUM_EPISODES=50
OUTPUT_FILE="policy_transfer_results.json"

echo "Cross-Simulator Policy Transfer - Quick Start"
echo "=============================================="
echo ""

# Test with MetaDrive conda env
echo "Testing adapter with MetaDrive environment..."
conda run -n metadrive python << 'EOF'
import sys
try:
    from metadrive.examples.scenefactory_adapter import SceneFactoryToMetaDriveAdapter
    import numpy as np
    
    adapter = SceneFactoryToMetaDriveAdapter(deterministic=True)
    print("✓ Adapter initialized successfully")
    
    # Quick test
    dummy_obs = np.random.randn(1929).astype(np.float32)
    md_obs = adapter.scenefactory_to_metadrive(dummy_obs)
    md_action = adapter.get_metadrive_expert_action(md_obs)
    sf_action = adapter.metadrive_to_scenefactory_action(md_action)
    print(f"✓ Data flow test passed: {dummy_obs.shape} → {md_obs.shape} → {md_action.shape} → {sf_action.shape}")
    
except Exception as e:
    print(f"✗ Error: {e}", file=sys.stderr)
    sys.exit(1)
EOF

echo ""
echo "✓ All checks passed!"
echo ""
echo "To run the full benchmark with SceneFactory:"
echo "  cd /home/yz8733/Github/isaac-rl"
echo "  conda run -n isaac-rl python benchmark_policy_transfer.py --num-episodes 50 --output results.json --latex"
echo ""
echo "The benchmark will:"
echo "  - Run 50 episodes of MetaDrive expert policy on SceneFactory"
echo "  - Measure success rate, collision rate, and rewards"
echo "  - Generate LaTeX table row for your paper"
echo "  - Save detailed results to results.json"
echo ""
