#!/bin/bash
#SBATCH --job-name=mpt7b-chat
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:1

echo "I AM ON "$(hostname) " running mpt 7b-chat"

cd /p/haicluster/llama/FastChat
source sc_venv_template/activate.sh


srun python3 -m fastchat.serve.model_worker \
     --controller http://haicluster1.fz-juelich.de:21001 \
     --port 31007 --worker http://$(hostname):31007 \
     --num-gpus 1 \
     --host 0.0.0.0 \
     --model-path /p/haicluster/llama/FastChat/models/mpt-7b-chat

#     --max-gpu-memory 22Gb \
