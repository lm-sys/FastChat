#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 \
python3 -m \
fastchat.serve.model_worker \
--model-path $2 \
--controller http://localhost:21001 \
--load-4bit \
--dtype float16 \
$3 $4 $5 $6 $7 $8 $9
