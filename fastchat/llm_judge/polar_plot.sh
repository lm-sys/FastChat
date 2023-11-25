#!/bin/bash

python polar_plot.py \
    --output_dir outputs/single_gpt4-turbo-judge \
    --model_judgment_fn data/mt_bench/model_judgment/gpt-4-1106-preview_single.jsonl
