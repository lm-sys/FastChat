#!/bin/bash

# Run with 'source scripts/serve-mps/start.sh'
    
for s in $(screen -ls|ggrep -o -P "\d+\.fastchat\.(.*)"); do screen -S $s -X stuff $'\003\003\003\003'; done;
sleep 1
for s in $(screen -ls|ggrep -o -P "\d+\.fastchat\.(.*)"); do screen -X -S $s quit; done;

sleep 3

if [[ "${1}" == "stop" ]]; then
    killall python3;
    exit 0;
fi

# screen -x fastchat.worker
# screen -x fastchat.controller
# screen -x fastchat.webserver

# for s in $(screen -ls|ggrep -o -P "\d+\.fastchat\.(.*)"); do screen -X -S $s quit; done;
#for s in $(screen -ls|ggrep -o -P "\d+\.fastchat\.(.*)"); do screen -S $s -X stuff $'\003\003\003\003'; done;
export CONDA_BASE_ENV_DIR="/Users/panayao/mambaforge";
export CONDA_ENV_NAME="ml";
export FASTCHAT_PARENT_DIR="/Users/panayao/Documents"; # Do not include trailing slash

screen -dmS fastchat.worker zsh -c "\
    sleep 0.5; \
    source ${CONDA_BASE_ENV_DIR}/bin/activate && \
    conda activate ${CONDA_ENV_NAME}; \
    ${FASTCHAT_PARENT_DIR}/FastChat/scripts/serve-mps/worker.7B+Vicuna_HF.sh; \
    exec zsh" && echo "Launched 'fastchat.worker'"
    
screen -dmS fastchat.controller zsh -c "\
    sleep 0.5; \
    source ${CONDA_BASE_ENV_DIR}/bin/activate && \
    conda activate ${CONDA_ENV_NAME}; \
    ${FASTCHAT_PARENT_DIR}/FastChat/scripts/serve-mps/controller.sh; \
    exec zsh" && echo "Launched 'fastchat.controller'"
    
echo "sleeping for 30 seconds to allow worker to bind to controller ..." && sleep 30
    
screen -dmS fastchat.webserver zsh -c "\
    sleep 0.5; \
    source ${CONDA_BASE_ENV_DIR}/bin/activate &&\
    conda activate ${CONDA_ENV_NAME}; \
    ${FASTCHAT_PARENT_DIR}/FastChat/scripts/serve-mps/webserver.sh; \
    exec zsh" && echo "Launched 'fastchat.webserver'"

alias fsctrl='screen -x fastchat.controller'
alias fsweb='screen -x fastchat.webserver'
alias fsworker='screen -x fastchat.worker'