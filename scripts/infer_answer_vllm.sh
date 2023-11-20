#!/bin/bash

cd /root/autodl-tmp/software/FastChat/fastchat/llm_judge

model_name=$1
model_id=$2
data_id=$3

python gen_model_answer.py --model-path $model_name --model-id $model_id --bench-name $data_id
