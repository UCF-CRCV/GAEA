#!/bin/bash

# Set the script directory
cd ..

BASE_DIR=$(pwd)

# Set the path to GAEA weights and GAEA-Bench images
MODEL_BASE="${BASE_DIR}/Qwen2.5-VL-7B-Instruct"
GAEA_WEIGHTS="${BASE_DIR}/merged/GAEA"
GAEA_BENCH="${BASE_DIR}/GAEA-Bench"

GAEA_BENCH_JSON="${BASE_DIR}/data/GAEA-Bench.json"

python inference/inference_mp16.py \
    --file_path $GAEA_BENCH_JSON \
    --img_root $GAEA_BENCH \
    --pretrained $GAEA_WEIGHTS \
    --model_base $MODEL_BASE \
    --save_dir "${BASE_DIR}/inference/predictions/gaea-gench/" \

python evaluations/gaea-bench/gpt_eval.py \
    --json_path "${BASE_DIR}/inference/predictions/gaea-gench/GAEA-predictions.json" \
    --outfolder "${BASE_DIR}/evaluations/gaea-bench/jsonl_batches/"

python evaluations/gaea-bench/upload.py \
    --input_dir "${BASE_DIR}/evaluations/gaea-bench/jsonl_batches/" \
    --output_dir "${BASE_DIR}/evaluations/gaea-bench/file_ids/" \
    --pattern "*.jsonl"

python evaluations/gaea-bench/submit.py \
    --input_dir "${BASE_DIR}/evaluations/gaea-bench/file_ids" \
    --output_dir "${BASE_DIR}/evaluations/gaea-bench/batch_ids/"

python evaluations/gaea-bench/retrieve.py \
    --input_dir "${BASE_DIR}/evaluations/gaea-bench/batch_ids/" \
    --output_dir "${BASE_DIR}/evaluations/gaea-bench/jsonl_responses" \
    --pattern "*.txt"

python evaluations/gaea-bench/calculate_scores.py \
    --response_dir "${BASE_DIR}/evaluations/gaea-bench/jsonl_responses" \
    --benchmark_file $GAEA_BENCH_JSON \
    --output_csv "${BASE_DIR}/evaluations/gaea-bench/output_csv"








