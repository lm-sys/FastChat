#!/bin/bash

local_path=$1
remote_path=$2

# This script is used to periodically copy local checkpoint to mounted storage
while true; do
    sleep 600
    local_last_ckpt=$(ls ${local_path} | grep -E '[0-9]+' | sort -t'-' -k1,1 -k2,2n | tail -1)
    remote_last_ckpt=$(ls ${remote_path} | grep -E '[0-9]+' | sort -t'-' -k1,1 -k2,2n | tail -1)
    echo "local_last_ckpt: ${local_last_ckpt}"
    echo "remote_last_ckpt: ${remote_last_ckpt}"
    if [ "${local_last_ckpt}" != "${remote_last_ckpt}" ]; then
        rsync -avz ${local_path}/${local_last_ckpt} ${remote_path}/
    fi
done
