#!/bin/bash

# Base directory paths
BASE_DIR="path/to/GAEA"
SCRIPT_DIR="$BASE_DIR/evaluations/distance"

# Change to the script's directory
cd ../../

# Create output directory if it doesn't exist
mkdir -p outs

# Define datasets to evaluate
datasets=(
    "im2gps im2gps.json"
    "im2gps3k im2gps3k.json"
    "yfcc4k yfcc4k.json"
    "gws15k gws15k.json"
    "yfcc26k yfcc26k.json"
)

# Evaluate GAEA on each dataset
for dataset in "${datasets[@]}"; do
    read -r name file <<< "$dataset"
    
    echo "Evaluations on $name for GAEA"
    python "${SCRIPT_DIR}/dist_acc.py" \
      --pred_path "${BASE_DIR}/GAEA/$file" \
      --outfolder "GAEA" \
      --outfile "$file"
done
