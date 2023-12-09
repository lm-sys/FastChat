#!/bin/bash
# A rather convenient script for spinning up models behind screens


# Variables
PROJECT_DIR="$(pwd)"
CONDA_ENV_NAME="fastchat" #

MODEL_PATH="HuggingFaceH4/zephyr-7b-beta" #beta is better than the alpha version, base model w/o quantization
MODEL_PATH="lmsys/vicuna-7b-v1.5" 

API_HOST="0.0.0.0"
API_PORT_NUMBER=8000


# init the screens
check_and_create_screen() {
    local SCREENNAME="$1"
    if screen -list | grep -q "$SCREENNAME"; then
        echo "Screen session '$SCREENNAME' exists. Doing nothing."
    else
        echo "Screen session '$SCREENNAME' not found. Creating..."
        screen -d -m -S "$SCREENNAME"
        echo "created!"
    fi
}

# convenience function for sending commands to named screens
send_cmd() {
    local SCREENNAME="$1"
    local CMD="$2"
    screen -DRRS $SCREENNAME -X stuff '$2 \r'
}

# hardcoded names, for baby api
SCREENNAMES=(
    "controller"
    "api"
    # Worker screens include the devices they are bound to, if 'd0' is only worker it has full GPU access
    "worker-d0"
    "worker-d1"
)

for screen in "${SCREENNAMES[@]}"; do
    check_and_create_screen "$screen"
    sleep 0.1
    # also activate the conda compute environment for these
    screen -DRRS "$screen" -X stuff "conda deactivate \r"
    screen -DRRS "$screen" -X stuff "conda activate $CONDA_ENV_NAME \r"
    
done


# Send Commmands on a per Screen Basis
screen -DRRS controller -X stuff  "python3 -m fastchat.serve.controller \r"

screen -DRRS worker-d0 -X stuff  "CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.model_worker --model-path $MODEL_PATH  --conv-template one_shot --limit-worker-concurrency 1  \r"
screen -DRRS worker-d1 -X stuff  "CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.model_worker --model-path $MODEL_PATH --port 21003 --worker-address http://localhost:21003  --conv-template one_shot --limit-worker-concurrency 1  \r"

screen -DRRS api -X stuff  "python3 -m fastchat.serve.openai_api_server --host $API_HOST --port $API_PORT_NUMBER \r"
