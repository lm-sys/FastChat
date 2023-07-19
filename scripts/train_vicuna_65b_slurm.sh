#!/bin/bash
#SBATCH --job-name=hao_65b_gpt4_0521 # create a short name for your job
#SBATCH --nodes=3
#SBATCH --gres=gpu:16      # number of gpus per node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=48:00:00     # total run time limit (HH:MM:SS)
#SBATCH --partition=high-profile
#SBATCH --error=/nfs/projects/mbzuai/ext_hao.zhang/hao/slurm_logs/job%J.%N.65b.err
#SBATCH --output=/nfs/projects/mbzuai/ext_hao.zhang/hao/slurm_logs/job%J.%N.65b.out
##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= " $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
# If you want to load things from your .bashrc profile, e.g. cuda drivers, singularity etc
cd /nfs/projects/mbzuai/ext_hao.zhang/
source ~/.bashrc
conda activate hao-env
cd hao/FastChat
free -g 2>&1
lscpu 2>&1
# ******************* These are read internally it seems ***********************************
# ******** Master port, address and world size MUST be passed as variables for DDP to work
export MASTER_PORT=20001
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "MASTER_PORT"=$MASTER_PORT
#echo "WORLD_SIZE="$WORLD_SIZE
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
#echo "MASTER_ADDR="$MASTER_ADDR
# ******************************************************************************************
echo "Run started at:- "
date
# Actual run of script
#srun python main.py # Use this if you have python in your environment
srun scripts/train_vicuna_65b_single_node_slurm.sh