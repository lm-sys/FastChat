DATA_PATH=../data/sharegpt4/sharegpt_gpt4_train.json
EVAL_DATA_PATH=../data/sharegpt4/sharegpt_gpt4_test.json
MODEL_NAME=<your model path>
OUTPUT_DIR=./checkpoints_qlora_13b_base
PATH_TO_DEEPSPEED_CONFIG=../playground/deepspeed_config_lora_s2.json

deepspeed ../fastchat/train/train_lora.py \
    --model_name_or_path $MODEL_NAME \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path $DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --bf16 True \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 3e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --q_lora True \
    --deepspeed $PATH_TO_DEEPSPEED_CONFIG \
    --gradient_checkpointing True \
    --flash_attn True
