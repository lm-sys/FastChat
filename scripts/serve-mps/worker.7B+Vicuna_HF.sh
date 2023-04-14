#!/bin/zsh

# Run with 'source scripts/serve-mps/worker.7B+Vicuna_HF.sh'

conda deactivate
conda activate ml

# Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

python3 -m fastchat.serve.model_worker \
	--model-path /Users/panayao/Documents/FastChat/LLaMA/hf/7B+Vicuna_HF \
	--device mps \
	--controller http://localhost:21001 \
	--port 31001 \
	--worker http://localhost:31001 \
	--limit-model-concurrency 8
