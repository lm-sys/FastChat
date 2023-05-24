#!/bin/bash
#SBATCH --job-name=test_comm  # create a short name for your job
#SBATCH --nodes=2
#SBATCH --gres=gpu:16      # number of gpus per node
#SBATCH --exclusive
#SBATCH --time=30-00:00:00     # total run time limit (HH:MM:SS)
#SBATCH --reservation=high-profile
#SBATCH --partition=high-profile
#SBATCH --error=/nfs/projects/mbzuai/ext_hao.zhang/hao/slurm_logs/job%J.%N.err
#SBATCH --output=/nfs/projects/mbzuai/ext_hao.zhang/hao/slurm_logs/job%J.%N.out

set -x
SLURM_GPUS_PER_TASK=16
ray stop

# __doc_head_address_start__

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi
# __doc_head_address_end__


cd /nfs/projects/mbzuai/ext_hao.zhang/
source ~/.bashrc
conda activate hao-env

# __doc_head_ray_start__
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-gpus "${SLURM_GPUS_PER_TASK}" --block &
# __doc_head_ray_end__

# __doc_worker_ray_start__
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-gpus "${SLURM_GPUS_PER_TASK}" --block &
    sleep 5
done

cd hao/alpa/benchmark/cupy
ray status
python -u profile_communication.py
