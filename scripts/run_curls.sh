#!/bin/bash

# 检查是否提供了 data_id 参数
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <data_id>"
    exit 1
fi

# 从脚本的第一个参数获取 data_id
DATA_ID=$1

# 发送请求，使用提供的 DATA_ID
curl -X POST -H "Content-Type: application/json" http://127.0.0.1:5004/generate -d "{\"model_name\":\"ZhipuAI/chatglm3-6b\",\"model_id\":\"chatglm\",\"data_id\":\"$DATA_ID\"}"
curl -X POST -H "Content-Type: application/json" http://127.0.0.1:5004/generate -d "{\"model_name\":\"ZhipuAI/chatglm2-6b\",\"model_id\":\"chatglm\",\"data_id\":\"$DATA_ID\"}"
curl -X POST -H "Content-Type: application/json" http://127.0.0.1:5004/generate -d "{\"model_name\":\"baichuan-inc/Baichuan2-7B-Chat\",\"model_id\":\"baichuan-chat\",\"data_id\":\"$DATA_ID\"}"
curl -X POST -H "Content-Type: application/json" http://127.0.0.1:5004/generate -d "{\"model_name\":\"qwen/Qwen-7B-Chat\",\"model_id\":\"qwen-chat\",\"data_id\":\"$DATA_ID\"}"
curl -X POST -H "Content-Type: application/json" http://127.0.0.1:5004/generate -d "{\"model_name\":\"Shanghai_AI_Laboratory/internlm-chat-7b\",\"model_id\":\"internlm-chat\",\"data_id\":\"$DATA_ID\",\"revision\":\"v1.0.3\"}"
curl -X POST -H "Content-Type: application/json" http://127.0.0.1:5004/generate -d "{\"model_name\":\"01ai/Yi-6B-Chat\",\"model_id\":\"Yi-34b-chat\",\"data_id\":\"$DATA_ID\"}"
curl -X POST -H "Content-Type: application/json" http://127.0.0.1:5004/judge -d "{\"data_id\":\"$DATA_ID\"}"
