MODEL_NAME="/home/ro834336/GAEA-Qwen-2.5VL/Qwen2.5-VL-7B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

cd ..
python src/merge_lora_weights.py \
    --model-path /home/ro834336/GAEA-Qwen-2.5VL/checkpoint-8100-LoRA \
    --model-base $MODEL_NAME  \
    --save-model-path /home/ro834336/GAEA-Qwen-2.5VL/merged/Full-Dataset-16-Rank-Halfway \
    --safe-serialization