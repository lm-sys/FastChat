#!/bin/bash
script_name=$(basename "$0")
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'
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
SHOW_COMMAND="False"

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
        --c)
            SHOW_COMMAND="True"
            shift 1
            ;;
        --command)
            SHOW_COMMAND="True"
            shift 1
            ;;
        *)
        echo "Invalid argument: $1"
        exit 1
        ;;
    esac
done

if [[ "$SHOW_COMMAND" == "True" ]]; then
    echo -e "${YELLOW}  - Dry Run Command ${NC}"
    echo -e "${BLUE}torchrun \\
        --nproc_per_node=4 \\
        --master_port=20001 \\
        fastchat/train/alpaca_train.py \\
        --model_name_or_path ${RED}"$MODEL_PATH"${NC} \\
        --data_path ${RED}"$DATA_PATH"${NC} \\
        --bf16 ${RED}"$BF16"${NC} \\
        --output_dir ${RED}"$OUTPUT_DIR"${NC} \\
        --num_train_epochs ${RED}"$NUM_EPOCHS"${NC} \\
        --per_device_train_batch_size ${RED}"$PER_DEVICE_TRAIN_BATCH_SIZE"${NC} \\
        --per_device_eval_batch_size ${RED}"$PER_DEVICE_EVAL_BATCH_SIZE"${NC} \\
        --gradient_accumulation_steps ${RED}"$GRADIENT_ACCUMULATION_STEPS"${NC} \\
        --evaluation_strategy ${RED}"$EVALUATION_STRATEGY"${NC} \\
        --save_strategy ${RED}"$SAVE_STRATEGY"${NC} \\
        --save_steps ${RED}"$SAVE_STEPS"${NC} \\
        --save_total_limit ${RED}"$SAVE_TOTAL_LIMIT"${NC} \\
        --learning_rate ${RED}"$LEARNING_RATE"${NC} \\
        --weight_decay ${RED}"$WEIGHT_DECAY"${NC} \\
        --warmup_ratio ${RED}"$WARMUP_RATIO"${NC} \\
        --lr_scheduler_type ${RED}"$LR_SCHEDULER_TYPE"${NC} \\
        --logging_steps ${RED}"$LOGGING_STEPS"${NC} \\
        --fsdp ${RED}"$FSDP"${NC} \\
        --fsdp_transformer_layer_cls_to_wrap ${RED}"$FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP"${NC} \\
        --tf32 ${RED}"$TF32"${NC} \\
        --gradient_checkpointing ${RED}"$GRADIENT_CHECKPOINTING"${NC}"
else
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
fi