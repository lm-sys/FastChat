### Fine-tuning FastChat-T5
You can use the following command to train FastChat-T5 with 4 x A100 (40GB).
```bash
torchrun --nproc_per_node=4 --master_port=9778 fastchat/train/train_flant5.py \
    --model_name_or_path google/flan-t5-xl \
    --data_path ./data/dummy_conversation.json \
    --bf16 True \
    --output_dir ./checkpoints_flant5_3b \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap T5Block \
    --tf32 True \
    --model_max_length 2048 \
    --preprocessed_path ./preprocessed_data/processed.json \
    --gradient_checkpointing True 
```

After training, please use our post-processing [function](https://github.com/lm-sys/FastChat/blob/55051ad0f23fef5eeecbda14a2e3e128ffcb2a98/fastchat/utils.py#L166-L185) to update the saved model weight. Additional discussions can be found [here](https://github.com/lm-sys/FastChat/issues/643).

### Fine-tuning using (Q)LoRA
You can use the following command to train Vicuna-7B using QLoRA using ZeRO2. Note that ZeRO3 is not currently supported with QLoRA but ZeRO3 does support LoRA, which has a reference configuraiton under playground/deepspeed_config_s3.json. To use QLoRA, you must have bitsandbytes>=0.39.0 and transformers>=4.30.0 installed.
```bash
deepspeed fastchat/train/train_lora.py \
    --model_name_or_path ~/model_weights/llama-7b \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path ./data/dummy_conversation.json \
    --bf16 True \
    --output_dir ./checkpoints \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --q_lora True \
    --deepspeed playground/deepspeed_config_s2.json \
```

For T5-XL or XXL

```bash
deepspeed fastchat/train/train_lora_t5.py \
        --model_name_or_path google/flan-t5-xl    \
        --data_path ./data/dummy_conversation.json \
        --bf16 True \
        --output_dir ./checkpoints_flant5_3b \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1  \
        --gradient_accumulation_steps 4  \
        --evaluation_strategy "no"  \
        --save_strategy "steps"  \
        --save_steps 300 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0.     \
        --warmup_ratio 0.03    \
        --lr_scheduler_type "cosine"   \
        --logging_steps 1 \
        --model_max_length 2048    \
        --preprocessed_path ./preprocessed_data/processed.json \
        --gradient_checkpointing True \
        --q_lora True     \
        --deepspeed playground/deepspeed_config_s2.json
        
```

For more dataset format support

- file format support: json, jsonl, csv, tsv
- data format support: vicuna, alpaca, chip2, self-instruct, hh-rlhf, oasst1, input-output

```bash
deepspeed --master_port 61000 fastchat/train/train_lora_plus.py \
    --model_name_or_path huggyllama/llama-7b  \
    --dataset "data/alpaca_data_cleaned_1000.json" \
    --dataset_format "alpaca" \
    --cache_dir /data0/huggingface/hub/ \
    --output_dir ./checkpoints \
    --num_train_epochs 5 \
    --fp16 False \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --flash_attn False \
    --xformers True \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 20  \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --q_lora False \
    --data_seed 42 \
    --do_train \
    --model_max_length 2048 \
    --source_max_len 2048 \
    --target_max_len 256 \
    --do_eval \
    --eval_dataset_size 100 \
    --max_eval_samples 1000 \
    --dataloader_num_workers 3 \
    --remove_unused_columns False \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --seed 0 \
    --report_to wandb \
    --deepspeed playground/deepspeed_config_s2.json
```