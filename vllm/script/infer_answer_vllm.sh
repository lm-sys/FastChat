#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python /ML-A100/home/tianyu/vllm_infer/vllm/evaluation/vllm_inference.py \
    --model-path "/ML-A100/home/tianyu/modelscope/qwen/Qwen-7B-Chat/" \
  --model-id  "qwen-7b-chat" \
  --max_token 512 \
  --data-path "/ML-A100/home/tianyu/vllm_infer/vllm/data_choice_sample100/" \
  --output-path "/ML-A100/home/tianyu/vllm_infer/vllm/output/model_answer/" \
  --card-id 0 \
  --tensor-parallel-size 1 \
  --identifier "Qwen"