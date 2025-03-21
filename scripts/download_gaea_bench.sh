#!/bin/bash
# This script downloads GAEA-Bench datasets from Huggingface
# and organizes the data into the specified directory structure.

# Set Python path
export PYTHONPATH=src:$PYTHONPATH

# Define working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_DIR/..

# Make the download script executable
chmod +x scripts/download_gaea.py

# ========================
# GAEA-Bench Dataset
# ========================
BENCH_DATASET="ucf-crcv/GAEA-Bench"  # Replace with actual dataset name

echo "======================================================="
echo "Starting download of GAEA-Bench dataset..."
echo "======================================================="
python scripts/download_gaea.py \
    --root_dir "GAEA-Bench" \
    --dataset_name "$BENCH_DATASET" \
    --split "test"

python scripts/download_json.py \
    --file_name "GAEA-Bench.json" \
    --base-url "https://huggingface.co/datasets/ucf-crcv/GAEA-Bench/resolve/main/" \
    --save-dir "data"
