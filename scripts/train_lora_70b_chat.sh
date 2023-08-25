TRAIN_DATA_PATH=../data/sharegpt_gpt4_train.json
EVAL_DATA_PATH=../data/sharegpt_gpt4_test.json
MODEL_NAME=<your model path>
OUTPUT_DIR=./checkpoints_lora_70b_chat
PATH_TO_DEEPSPEED_CONFIG=/workspace/zhouyou/FastChat/playground/deepspeed_config_lora_s3.json

deepspeed ../fastchat/train/train_lora.py \
    --model_name_or_path $MODEL_NAME \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path $TRAIN_DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 5 \
    --bf16 True \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --learning_rate 3e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --deepspeed $PATH_TO_DEEPSPEED_CONFIG \
    --gradient_checkpointing True \
    --flash_attn False
