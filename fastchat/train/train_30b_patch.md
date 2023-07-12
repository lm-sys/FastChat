## Apply the following patches in order to train 30B

1. Change [here](https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train.py#L241-L244) to:
```python
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=training_args.cache_dir,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
)
```

2. Add the following after [line 1498 of transformers/src/trainer.py](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L1498):
```python
if model.dtype == torch.float16:
    self.model = model = model.float()
```

#!/bin/bash
echo "NODE_RANK="$SLURM_NODEID
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE
python -m torch.distributed.run --nproc_per_node=16 --nnodes $SLURM_NNODES --node_rank=$SLURM_PROCID \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    longchat/train/pretrain/pretrain_xformer.py \
    --model_name_or_path /nfs/projects/mbzuai/ext_hao.zhang/dacheng/llama-7B-hf --data_path /nfs/projects/mbzuai/ext_hao.zhang/dacheng/book_partial.jsonl \
    --fp16 \
    --output_dir pretrain_llama_7B_8192 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_steps 100 \
    --lr_scheduler_type polynomial \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --begin 4300 \