#!/bin/bash

# Default values
MODEL_PATH="/home/haozhang/model_weights/hf-llama-7b"
DATA_PATH="/home/haozhang/datasets/alpaca_data.json"
BF16="True"
OUTPUT_DIR="output"
NUM_TRAIN_EPOCHS="3"
PER_DEVICE_TRAIN_BATCH_SIZE="4"
PER_DEVICE_EVAL_BATCH_SIZE="4"
GRADIENT_ACCUMULATION_STEPS="8"
EVALUATION_STRATEGY="no"
SAVE_STRATEGY="steps"
SAVE_STEPS="2000"
SAVE_TOTAL_LIMIT="1"
LEARNING_RATE="2e-5"
WEIGHT_DECAY="0."
WARMUP_RATIO="0.03"
LR_SCHEDULER_TYPE="cosine"
LOGGING_STEPS="1"
FSDP="full_shard auto_wrap"
FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP="LlamaDecoderLayer"
TF32="True"
GRADIENT_CHECKPOINTING="True"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name_or_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --bf16)
            BF16="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_train_epochs)
            NUM_TRAIN_EPOCHS="$2"
            shift 2
            ;;
        --per_device_train_batch_size)
            PER_DEVICE_TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --per_device_eval_batch_size)
            PER_DEVICE_EVAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --gradient_accumulation_steps)
            GRADIENT_ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        --evaluation_strategy)
            EVALUATION_STRATEGY="$2"
            shift 2
            ;;
        --save_strategy)
            SAVE_STRATEGY="$2"
            shift 2
            ;;
        --save_steps)
            SAVE_STEPS="$2"
            shift 2
            ;;
        --save_total_limit)
            SAVE_TOTAL_LIMIT="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --weight_decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --warmup_ratio)
            WARMUP_RATIO="$2"
            shift 2
            ;;
        --lr_scheduler_type)
            LR_SCHEDULER_TYPE="$2"
            shift 2
            ;;
        --logging_steps)
            LOGGING_STEPS="$2"
            shift 2
            ;;
        --fsdp)
            FSDP="$2"
            shift 2
            ;;
        --fsdp_transformer_layer_cls_to_wrap)
            FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP="$2"
            shift 2
            ;;
        --tf32)
            TF32="$2"
            shift 2
            ;;
        --gradient_checkpointing)
            GRADIENT_CHECKPOINTING="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

torchrun \
    --nproc_per_node=4 \
    --master_port=20001 \
    fastchat/train/alpaca_train.py \
    --model_name_or_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --bf16 "$BF16" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --evaluation_strategy "$EVALUATION_STRATEGY" \
    --save_strategy "$SAVE_STRATEGY" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --warmup_ratio "$WARMUP_RATIO" \
    --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
    --logging_steps "$LOGGING_STEPS" \
    --fsdp "$FSDP" \
    --fsdp_transformer_layer_cls_to_wrap "$FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP" \
    --tf32 "$TF32" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING"