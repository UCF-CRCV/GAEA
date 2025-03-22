#!/bin/bash

# Model configuration
MODEL_NAME="GAEA"
MODEL_BASE="Qwen/Qwen2.5-VL-7B-Instruct"
TEMPERATURE=0.2

# Set Python path
export PYTHONPATH=src:$PYTHONPATH

cd ..

BASE_DIR=$(pwd)

# Paths configuration
PRED_FOLDER="${BASE_DIR}/location_predictions/${MODEL_NAME}"
GT_FOLDER="${BASE_DIR}/evaluations/city-country/ground-truth"
EVAL_DIR="${BASE_DIR}/evaluations/city-country"
JSONLS_DIR="${EVAL_DIR}/jsonls"
OUTPUTS_DIR="${EVAL_DIR}/outputs"
RESULTS_DIR="${EVAL_DIR}/results"

# Define datasets and their paths with corresponding image folders if needed
declare -A DATASETS=(
    ["cityguessr"]="path/to/cityguessr/images"
    ["dollarstreet"]="path/to/dollarstreet/images"
    ["geode"]="path/to/geode/images"
)

# Define CSV data paths which contain GT city and country names
declare -A CSV_DATA=(
    ["cityguessr"]="${GT_FOLDER}/cityguessr_citygt.csv"
    ["dollarstreet"]="${GT_FOLDER}/dollarstreet_citygt.csv"
    ["geode"]="${GT_FOLDER}/geode_citygt.csv"
)

# Create directories if they don't exist
mkdir -p "${PRED_FOLDER}" "${JSONLS_DIR}" "${OUTPUTS_DIR}" "${RESULTS_DIR}"

# 1. Run inference for each dataset
echo "STEP 1: Running inference for all datasets using model: ${MODEL_NAME}"
for dataset_name in "${!DATASETS[@]}"; do
    dataset_path="${DATASETS[$dataset_name]}"
    csv_path="${CSV_DATA[$dataset_name]}"
    output_file="${PRED_FOLDER}/${dataset_name}.json"
    
    echo "Processing dataset: ${dataset_name}"
    echo "Dataset path: ${dataset_path}"
    echo "CSV data: ${csv_path}"
    echo "Output file: ${output_file}"
    
    python inference/location_inference.py \
        --pretrained "merged/${MODEL_NAME}" \
        --image_dir "${dataset_path}" \
        --outpath "${output_file}" \
        --save_dir "${MODEL_NAME}" \
        --csv_data "${csv_path}" \
        --temperature ${TEMPERATURE}
    
    echo "Completed processing ${dataset_name}"
    echo "--------------------------------------"
done

echo "All inference tasks completed!"
echo ""

# 2. Create evaluation batches
echo "STEP 2: Creating evaluation batches for city and country prediction"
python ${EVAL_DIR}/batchFile_city_eval.py \
    --pred_folder "${PRED_FOLDER}" \
    --gt_folder "${GT_FOLDER}" \
    --output_folder "${JSONLS_DIR}" \
    --model_name "${MODEL_NAME}"

python ${EVAL_DIR}/batchFile_country_eval.py \
    --pred_folder "${PRED_FOLDER}" \
    --gt_folder "${GT_FOLDER}" \
    --output_folder "${JSONLS_DIR}" \
    --model_name "${MODEL_NAME}"

echo "Evaluation batches created!"
echo ""

# 3. Upload files to Azure OpenAI
echo "STEP 3: Uploading evaluation batches to Azure OpenAI"
python ${EVAL_DIR}/submit_file.py \
    --input_dir "${JSONLS_DIR}" \
    --output_file "${EVAL_DIR}/file_ids.json"

echo "Files uploaded!"
echo ""

# 4. Submit batch jobs
echo "STEP 4: Submitting batch jobs to Azure OpenAI"
python ${EVAL_DIR}/submit_job.py \
    --file_ids_json "${EVAL_DIR}/file_ids.json" \
    --output_file "${EVAL_DIR}/batch_ids.json"

echo "Batch jobs submitted!"
echo ""

# 5. Retrieve results (with polling until complete)
echo "STEP 5: Retrieving results for city evaluation"
python ${EVAL_DIR}/retrieve.py \
    --batch_ids_json "${EVAL_DIR}/batch_ids.json" \
    --output_dir "${OUTPUTS_DIR}" \
    --eval_type "city" \
    --poll_interval 60

echo "STEP 5: Retrieving results for country evaluation"
python ${EVAL_DIR}/retrieve.py \
    --batch_ids_json "${EVAL_DIR}/batch_ids.json" \
    --output_dir "${OUTPUTS_DIR}" \
    --eval_type "country" \
    --poll_interval 60

echo "Results retrieved!"
echo ""

# 6. Calculate accuracy
echo "STEP 6: Calculating city and country prediction accuracy"
python ${EVAL_DIR}/cc_acc.py \
    --city_file "${OUTPUTS_DIR}/GAEA_city.json" \
    --country_file "${OUTPUTS_DIR}/GAEA_country.json" \
    --model_name "${MODEL_NAME}" \
    --output_file "${RESULTS_DIR}/${MODEL_NAME}_results.csv"

echo "Evaluation completed! Results saved to ${RESULTS_DIR}/${MODEL_NAME}_results.csv"

    
