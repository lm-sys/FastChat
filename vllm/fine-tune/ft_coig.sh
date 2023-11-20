hostfile=""
model="Baichuan2-13B-Base"
checkpoint_dir="base_13b"
deepspeed --hostfile=$hostfile fine_tune_COIG.py  \
    --report_to "none" \
    --data_path "/cpfs/29cd2992fe666f2a/user/huangwenhao/xw/Humpback-CH/data/COIG/human_value_alignment_instructions_part1.json" \
    --model_name_or_path "/cpfs/29cd2992fe666f2a/shared/public/baichuan_model/$model" \
    --output_dir "/cpfs/29cd2992fe666f2a/user/huangwenhao/xw/Humpback-CH/checkpoint/$checkpoint_dir" \
    --use_lora False \
    --model_max_length 512 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --learning_rate 2e-5 \
    --lr_scheduler_type constant \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --deepspeed ds_config.json \
    --bf16 True \
    --tf32 True