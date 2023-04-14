#!/bin/zsh

# Run with 'source webserver.sh'

conda deactivate
conda activate ml

# Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

python3 -m fastchat.serve.gradio_web_server \
    --concurrency-count 20 \
    --model-list-mode reload
