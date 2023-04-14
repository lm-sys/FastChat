#!/bin/zsh

# Run with 'source run.13B.sh'

conda deactivate 
conda activate ml

# Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

python3 -m fastchat.serve.cli \
	--model-name /Users/panayao/Documents/FastChat/LLaMA/hf/13B+Vicuna_HF \
	--device mps \
	--style rich \
	--temperature 0.35 \
	--max-new-tokens 1024 \
	--top-p 0.5 \
	--top-k 0.5
