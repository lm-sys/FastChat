"""
Get stats of a dataset.

Usage: python3 -m fastchat.data.get_stats --in sharegpt.json
"""

import argparse
import json

from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np


def compute_avg_turns(content):
    turns = []

    for c in content:
        turns.append(len(c["conversations"]) // 2)

    return np.mean(turns)


def compute_avg_response_length(content, tokenizer):
    res_lens = []

    for c in content:
        for i in range(len(c["conversations"]) // 2):
            v = c["conversations"][i * 2 + 1]["value"]
            res_lens.append(len(tokenizer.tokenize(v)))

    return np.mean(res_lens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str)
    parser.add_argument("--model-path", type=str)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    content = json.load(open(args.in_file, "r"))

    avg_turns = compute_avg_turns(content)
    avg_res_len = compute_avg_response_length(content, tokenizer)

    print(f"#sequence: {len(content)}")
    print(f"avg. turns: {avg_turns:.2f}")
    print(f"avg. response length: {avg_res_len:.2f}")
