#!/bin/bash

# Model configuration
MODEL_NAME="merged/GAEA"
BASE_DIR=$(pwd)
TEMPERATURE=0.2

# Set Python path
export PYTHONPATH=src:$PYTHONPATH

cd ../..
# Define datasets and their paths
declare -A DATASETS=(
    ["im2gps"]="path/to/im2gps"
    ["im2gps3k"]="path/to/im2gps3k"
    ["yfcc4k"]="path/to/yfcc4k"
    ["gws15k"]="path/to/gws15k"
    ["yfcc26k"]="path/to/yfcc26k"
)

# Run inference for each dataset in a loop
echo "Starting inference for all datasets using model: $MODEL_NAME"
for dataset_name in "${!DATASETS[@]}"; do
    dataset_path="${DATASETS[$dataset_name]}"
    output_file="${dataset_name}.json"
    
    echo "Processing dataset: $dataset_name"
    echo "Dataset path: $dataset_path"
    echo "Output file: $output_file"
    
    python location_inference.py \
        --pretrained "$MODEL_NAME" \
        --image_dir "$dataset_path" \
        --outpath "$output_file" \
        --save_dir "$MODEL_NAME" \
        --temperature $TEMPERATURE
    
    echo "Completed processing $dataset_name"
    echo "--------------------------------------"
done

echo "All inference tasks completed!"