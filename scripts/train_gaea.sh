#!/bin/bash


# If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together
cd ..

BASE_DIR=$(pwd)

MODEL_NAME="${BASE_DIR}/Qwen/Qwen2.5-VL-7B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=16
NUM_DEVICES=2
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))


echo Gradient Accumulation set to $GRAD_ACCUM_STEPS

GAEA_1-6M_FOLDER="${BASE_DIR}/GAEA-1.6M"

MP16_FOLDER="${GAEA_1-6M_FOLDER}/MP-16"
GLDV2_FOLDER="${GAEA_1-6M_FOLDER}/GLDv2"
CITY_GUESSR_FOLDER="${GAEA_1-6M_FOLDER}/CityGuessr"

DATA_PATH="${BASE_DIR}/data/GAEA-Train.json"

FT_MODEL_NAME="GAEA-LoRA"

deepspeed src/training/train.py \
    --lora_enable True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path $DATA_PATH \
    --image_folder_mp16 $MP16_FOLDER \
    --image_folder_gldv2 $GLDV2_FOLDER \
    --image_folder_cityguessr $CITY_GUESSR_FOLDER \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --tune_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "checkpoint/$FT_MODEL_NAME" \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --min_pixels $((256 * 28 * 28)) \
    --max_pixels $((1280 * 28 * 28)) \
    --learning_rate 1e-4 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 4000 \
    --save_total_limit 10 \
    --dataloader_num_workers 2 \
    --finetune_liger_norm False

#Merge the LoRA weights
python src/merge_lora_weights.py \
    --model-path checkpoint/$FT_MODEL_NAME \
    --model-base $MODEL_NAME  \
    --save-model-path merged/GAEA \
    --safe-serialization

# Copy necessary files from original model to merged model directory
echo "Copying additional necessary files from base model..."
python -c "
import os
import shutil

# Download the base model
model_path = '$MODEL_NAME'
merged_path = 'merged/GAEA'

# Files to copy (modify this list as needed)
files_to_copy = [
    'added_tokens.json',
    'chat_template.json',
    'preprocessor_config.json',
    'special_tokens_map.json',
    'tokenizer_config.json',
    'tokenizer.json',
    'vocab.json'
]

# Copy each file if it exists
for file in files_to_copy:
    src = os.path.join(model_path, file)
    dst = os.path.join(merged_path, file)
    if os.path.exists(src) and not os.path.exists(dst):
        print(f'Copying {file}...')
        shutil.copy(src, dst)
    elif not os.path.exists(src):
        print(f'Warning: {file} not found in original model')

print('Done copying files')
"

