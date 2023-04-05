#!/bin/bash

local_path=$1
remote_path=$2
MAX_NUM_CKPT=3

# This script is used to periodically copy local checkpoint to mounted storage
while true; do
    local_last_ckpt=$(ls ${local_path} | grep checkpoint- | grep -E '[0-9]+' | sort -t'-' -k1,1 -k2,2n | tail -1)
    remote_last_ckpt=$(ls ${remote_path} | grep checkpoint- | grep -E '[0-9]+' | sort -t'-' -k1,1 -k2,2n | tail -1)
    echo "local_last_ckpt: ${local_last_ckpt}"
    echo "remote_last_ckpt: ${remote_last_ckpt}"
    if [ "${local_last_ckpt}" != "${remote_last_ckpt}" ]; then
        mkdir -p ${remote_path}/${local_last_ckpt}
        gsutil -m rsync -r ${local_path}/${local_last_ckpt}/ ${remote_path}/${local_last_ckpt}

        # Keep only the last MAX_NUM_CKPT checkpoints
        num_local_ckpt=$(ls ${local_path} | grep checkpoint- | wc -l)
        echo "num_local_ckpt: ${num_local_ckpt}"
        if [ ${num_local_ckpt} -gt $MAX_NUM_CKPT ]; then
            for ckpt in $(ls ${local_path} | grep checkpoint- | grep -E '[0-9]+' | sort -t'-' -k1,1 -k2,2n | head -n-${MAX_NUM_CKPT}); do
                rm -rf ${local_path}/${ckpt}
            done
        fi
    fi

    sleep 600
done
