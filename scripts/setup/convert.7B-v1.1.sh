#!/bin/zsh

# Run with 'source scripts/setup/convert.7B.sh'

conda deactivate 
conda activate ml

python3 -m fastchat.model.apply_delta \
	--base decapoda-research/llama-7b-hf \
	--target /Users/panayao/Documents/FastChat/LLaMA/hf/7B+Vicuna_HF_v1.1 \
	--delta lmsys/vicuna-7b-delta-v1.1
