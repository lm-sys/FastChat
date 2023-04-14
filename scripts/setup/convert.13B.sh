#!/bin/zsh

# Run with 'source scripts/setup/convert.13B.sh'

conda deactivate 
conda activate ml

python3 -m fastchat.model.apply_delta \
	--base decapoda-research/llama-13b-hf \
	--target /Users/panayao/Documents/FastChat/LLaMA/hf/13B+Vicuna_HF \
	--delta lmsys/vicuna-13b-delta-v0
