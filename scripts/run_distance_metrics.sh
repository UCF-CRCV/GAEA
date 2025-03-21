#!/bin/bash

# Define model and output paths
MODEL_NAME="merged/GAEA"
OUTFOLDER="GAEA"

# Set Python path
export PYTHONPATH=src:$PYTHONPATH

cd ..

BASE_DIR=$(pwd)

# Function to run inference
run_inference() {
    python inference/location_inference.py \
        --pretrained $MODEL_NAME \
        --image_dir "$1" --outpath "$2" --save_dir "evaluations/distance/predictions/${OUTFOLDER}" --temperature 0.2
}

# Function to evaluate inference
evaluate_inference() {
    python evaluations/distance/geocode.py \
        --pred_path "${BASE_DIR}/evaluations/distance/predictions/${OUTFOLDER}/$1" \
        --outfolder "${BASE_DIR}/evaluations/distance/gecodings/${OUTFOLDER}/" \
        --outfile "$1"
}

# Check if SLURM is available
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Running on SLURM environment."
    # SLURM-specific settings
    # (You can add specific SLURM settings here if needed)
    # We recommend using at least one 48GB GPU with flash attention enabled
else
    echo "Running on non-SLURM environment."
fi

# Run inference and evaluation for each dataset
datasets=(
    "path/to/im2gps im2gps.json"
    "path/to/im2gps3ktest im2gps3k.json"
    "path/to/yfcc4k yfcc4k.json"
    "path/to/gws15k gws15k.json"
    "path/to/yfcc26k yfcc26k.json"
)

# Note: Running evaluations concurrently in the background is suitable for systems with ample memory.
# If memory is limited, consider running the two tasks separately for each dataset to avoid memory constraints 
# (scripts found in scripts/separate_tasks)
for dataset in "${datasets[@]}"; do
    read -r video_dir outpath <<< "$dataset"
    
    echo "Running inference on $video_dir"
    run_inference "$video_dir" "$outpath"
    
    # Start evaluation in the background
    echo "Starting evaluation for $outpath in the background"
    evaluate_inference "${outpath}" &
done

# Wait for all background jobs to finish
wait

declare -A GT_DATA=(
    ["im2gps.json"]="path/to/im2gps_citygt.csv"
    ["im2gps3k.json"]="path/to/im2gps3k_citygt.csv"
    ["yfcc4k.json"]="path/to/yfcc4k_citygt.csv"
    ["yfcc26k.json"]="path/to/yfcc26k_citygt.csv"
    ["gws15k.json"]="path/to/gws15k_citygt.csv"
)

# run dist_acc_geocoded in a loop for all distance metrics
for dataset in "${datasets[@]}"; do
    read -r video_dir outpath <<< "$dataset"
    dataset_gt="${GT_DATA[$outpath]}"
    echo "Running dist_acc_geocoded for $outpath"
    python evaluations/distance/dist_acc.py \
      --pred_path "${BASE_DIR}/evaluations/distance/gecodings/${OUTFOLDER}/${outpath}" \
      --gt_path $dataset_gt
done

