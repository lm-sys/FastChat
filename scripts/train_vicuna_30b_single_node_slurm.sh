#!/bin/bash
echo "NODE_RANK="$SLURM_NODEID
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE
python -m torch.distributed.run --nproc_per_node=16 --nnodes $SLURM_NNODES --node_rank=$SLURM_PROCID \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    fastchat/train/train_xformer.py \
    --model_name_or_path /nfs/projects/mbzuai/ext_hao.zhang/hao/dataset/llama-30b \
    --data_path /nfs/projects/mbzuai/ext_hao.zhang/hao/dataset/sharegpt_20230422_clean_lang_split_identity.json \
    --fp16 True \
    --output_dir vicuna_30b_sharegpt_20230422_32GPUs \
    --max_steps 6400 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 3000 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 4 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_steps 20 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True