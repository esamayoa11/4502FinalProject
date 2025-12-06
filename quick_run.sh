#!/bin/bash
# quick_run.sh
# Run full pipeline: preprocess + train both films

# Exit on error
set -e

echo "=== Step 1: Preprocessing images ==="
python src/preprocess.py

echo "=== Step 2: Training renderer on both films ==="
python src/train.py

echo "=== Step 3: Done! Results saved in 'results/' folder ==="