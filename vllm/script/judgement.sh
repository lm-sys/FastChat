#!/bin/bash

python /home/Userlist/madehua/code/vllm/evaluation/llm_judge/gen_judgment.py \
    --api_key "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" \
    --bench-name  "single_turn" \
    --judge-model "gpt-3.5-turbo" \
    --mode "single" \
    --question_file_name "questions.json" \
    --begin 2000 \
    --end 2060 \
    --model-list "baichaun-13b-chat" "chatglm2"