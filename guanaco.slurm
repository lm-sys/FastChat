#!/bin/bash
#SBATCH --job-name=guanaco-65b-merged
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:7


echo "I AM ON "$(hostname) " running guanaco-65b with 7 gpus"

cd /p/haicluster/llama/FastChat
source sc_venv_template/activate.sh


srun python3 -m fastchat.serve.model_worker \
     --controller http://haicluster1.fz-juelich.de:21001 \
     --port 31002 --worker http://$(hostname):31002 \
     --num-gpus 7 \
     --host 0.0.0.0 \
     --model-path /p/haicluster/llama/FastChat/models/guanaco-65b-merged 
