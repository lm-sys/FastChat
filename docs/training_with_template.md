### Fine-tuning Vicuna-7B with template

You can use the following command to train Mistral-7B with template.
"system" field in the training JSON can be different for each conversation.
Conversations with no "system" field will use the default system prompt defined in the template.
```bash
torchrun --nproc_per_node=2 --master_port=20001 fastchat/train/train_with_template.py \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --data_path data/dummy_conversation_with_system.json \
    --bf16 True \
    --output_dir mistral-7b-0103 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True
```
