#!/bin/bash
# Quick start script for FYP Machine Learning Pipeline
# Usage: ./quick_start.sh

set -e

# Change to the correct directory
cd "$(dirname "$0")"

echo "==================================="
echo "FYP Machine Learning Pipeline"
echo "==================================="
echo ""
echo "Working directory: $(pwd)"
echo ""

# Show menu
echo "What would you like to do?"
echo ""
echo "1. Train model"
echo "2. Evaluate model"
echo "3. Extract features from videos"
echo "4. Run real-time demo"
echo "5. Run tests"
echo "6. Process CLARE data"
echo ""
read -p "Enter choice (1-6): " choice

case $choice in
    1)
        echo "Training model..."
        python3 -m src.cle.train.train \
            --in data/processed/features.csv \
            --out models/ \
            --config configs/default.yaml
        ;;
    2)
        echo "Evaluating model..."
        python3 -m src.cle.train.eval \
            --in data/processed/features.csv \
            --models models/ \
            --report reports/eval.json \
            --config configs/default.yaml
        ;;
    3)
        echo "Extracting features..."
        python3 -m src.cle.extract.pipeline_offline \
            --manifest data/raw/manifest.csv \
            --out data/processed/features.csv \
            --config configs/default.yaml
        ;;
    4)
        echo "Starting real-time demo..."
        python3 -m src.cle.extract.pipeline_realtime \
            --models models/ \
            --config configs/default.yaml
        ;;
    5)
        echo "Running tests..."
        python3 -m pytest tests/ -v
        ;;
    6)
        echo "Processing CLARE data..."
        python3 scripts/process_clare_data.py \
            --clare_dir data/CLARE \
            --output data/processed/clare_features.csv
        ;;
    *)
        echo "Invalid choice!"
        exit 1
        ;;
esac

echo ""
echo "Done!"
