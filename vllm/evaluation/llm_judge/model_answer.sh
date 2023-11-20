#!/bin/bash

code_path="/home/Userlist/madehua/code/fc"
model_path="/home/Userlist/madehua/model"
export CUDA_VISIBLE_DEVICES="2"
model_name="Baichuan2-7B-Chat"
# 执行Python脚本并传递参数
python gen_model_answer.py \
  --model-path "$model_path/$model_name/" \
  --model-id  "llama-2" \
  --model-name "$model_name" \
  --max-new-token 512 \
  --bench-name "single_turn"
