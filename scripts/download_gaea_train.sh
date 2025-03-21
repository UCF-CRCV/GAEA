#!/bin/bash
# This script downloads GAEA-1.6M dataset from Huggingface
# and organizes the data into the specified directory structure.

# Set Python path
export PYTHONPATH=src:$PYTHONPATH

# Define working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_DIR/..

# Make the download script executable
chmod +x download_gaea.py

# ========================
# GAEA-Train Dataset
# ========================
TRAIN_DATASET="ucf-crcv/GAEA-Train"  # Replace with actual dataset name

echo "======================================================="
echo "Starting download of GAEA-Train dataset..."
echo "======================================================="
python download_gaea.py \
    --root_dir 'GAEA-1.6M' \
    --dataset_name "$TRAIN_DATASET" \
    --split "train"

python scripts/download_json.py \
    --file_name "GAEA-Train.json" \
    --base-url "https://huggingface.co/datasets/ucf-crcv/GAEA-Train/resolve/main/" \
    --save-dir "data"
