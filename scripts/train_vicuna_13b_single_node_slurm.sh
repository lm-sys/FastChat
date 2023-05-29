#!/bin/bash
echo "NODE_RANK="$SLURM_NODEID
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE
python -m torch.distributed.run --nproc_per_node=16 --nnodes $SLURM_NNODES --node_rank=$SLURM_PROCID \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    fastchat/train/train_xformer.py \
    --model_name_or_path /nfs/projects/mbzuai/ext_hao.zhang/hao/dataset/llama-13b \
    --data_path /nfs/projects/mbzuai/ext_hao.zhang/hao/dataset/sharegpt_20230515_clean_lang_split_identity_v2.json \
    --fp16 True \
    --output_dir vicuna_13b_full_sharegpt_20230515_v2_32GPU \
    --num_train_epochs 3 \
    --max_steps 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 8 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True
