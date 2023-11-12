
# MASTER_ADDR=${MASTER_ADDR:-localhost}
# MASTER_PORT=${MASTER_PORT:-23456}
# NNODES=${NODE_NUM:-1}
# NODE_RANK=${RANK:-0}
# GPUS_PER_NODE=${GPUS_NUM_PER_NODE:-$(nvidia-smi -L | wc -l)}
# DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"


# if [ "${K8S_JOB_TYPE}" == "notebook" ]; then
#     # 【无需修改】Notebook 环境存储根路径暂时固定
#     ROOT_DIR=/home/tione/notebook
# else
#     # 【自定义修改】任务式建模的训练任务中的存储根路径，需与启动任务时指定的路径一致
#     ROOT_DIR=/home/tione/notebook
# fi

# source ~/.zshrc
# wait 
# which python

DATA_PATH=/home/tione/notebook/dongwu/tg/domain/v5/yanbao/data/edb_recall/edb_recall_train_v1.json@@\
/home/tione/notebook/dongwu/tg/domain/v5/yanbao/data/llmgen/llmgen_v2.2/revise/train_all.json@@\
/home/tione/notebook/dongwu/tg/domain/v5/yanbao/data/metric_choose/metric_match_train_v1.json@@\
/home/tione/notebook/dongwu/tg/domain/v5/yanbao/data/outline/all_train_v2.json@@\
/home/tione/notebook/dongwu/tg/domain/v5/yanbao/data/qagen/qagen_v2/all_train_v2.json@@\
/home/tione/notebook/dongwu/tg/domain/v5/yanbao/data/sharegpt_20230621_zh_clean.json@@\
/home/tione/notebook/dongwu/tg/domain/v5/yanbao/data/llmgen/query_merge/query_merge.json

VALID_PATH=/home/tione/notebook/dongwu/tg/domain/v5/yanbao/data/metric_choose/metric_match_valid_v1.json

OUTPUT_DIR=/home/tione/notebook/dongwu/tg/domain/v5/model_hubs/my_checkpoint/yi_test
LOGGING_DIR=/home/tione/notebook/dongwu/tg/domain/v5/model_hubs/my_checkpoint/tensorboard/yi_test

mkdir -p $OUTPUT_DIR
mkdir -p $LOGGING_DIR
# export NCCL_DEBUG=WARN

export CUDA_VISIBEL_DEVICES=0
# --include='localhost:0,4,5,6,7'
deepspeed --master_port 20001 \
    /home/tione/notebook/dongwu/tg/domain/v5/open_repo/FastChat/fschat_scripts/train_v2.py \
    --model_name_or_path /home/tione/notebook/dongwu/tg/domain/v5/model_hubs/my_checkpoint/13b_zh_allstep_v2.3/checkpoint-240 \
    --tokenizer_path /home/tione/notebook/dongwu/tg/domain/v5/model_hubs/my_checkpoint/13b_zh_allstep_v2.3/checkpoint-240 \
    --data_path $DATA_PATH \
    --valid_path $VALID_PATH \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --eval_steps 10 \
    --save_strategy "steps" \
    --save_steps 40 \
    --save_total_limit 1 \
    --learning_rate 3e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --logging_dir $LOGGING_DIR \
    --deepspeed /home/tione/notebook/dongwu/tg/domain/v5/open_repo/FastChat/fschat_scripts/zero3_offload_config.json \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to 'none' 
#     --load_best_model_at_end True \
    #--optim adamw_apex_fused
