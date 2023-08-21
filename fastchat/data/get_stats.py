"""
Get stats of a dataset.

Usage: python3 -m fastchat.data.get_stats --in sharegpt.json
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

K = 1e3
M = 1e6


def tokenize_one_sample(c):
    for i in range(len(c["conversations"])):
        v = c["conversations"][i]["value"]
        c["conversations"][i]["value"] = tokenizer.tokenize(v)
    return c


def tokenize_dataset(content):
    processed = []
    with ProcessPoolExecutor() as executor:
        for result in tqdm(
            executor.map(tokenize_one_sample, content), total=len(content)
        ):
            processed.append(result)

    return processed


def compute_stats(content):
    sample_lens = []
    sample_turns = []
    prompt_lens = []
    res_lens = []

    for c in content:
        sample_len = 0
        sample_turns.append(len(c["conversations"]) // 2)
        for i in range(len(c["conversations"]) // 2):
            p = c["conversations"][i * 2]["value"]
            r = c["conversations"][i * 2 + 1]["value"]

            turn_len = len(p) + len(r)
            sample_len += turn_len
            prompt_lens.append(len(p))
            res_lens.append(len(r))
        sample_lens.append(sample_len)

    return sample_lens, sample_turns, prompt_lens, res_lens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str)
    parser.add_argument(
        "--model-name-or-path", type=str, default="meta-llama/Llama-2-7b-chat-hf"
    )
    args = parser.parse_args()

    content = json.load(open(args.in_file, "r"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    content = tokenize_dataset(content)

    sample_lens, sample_turns, prompt_lens, res_lens = compute_stats(content)
    print(f"#sequence: {len(content)/K:.2f} K")
    print(f"#tokens: {np.sum(sample_lens)/M:.2f} M")
    print(f"avg. turns: {np.mean(sample_turns):.2f}")
    print(f"avg. prompt length: {np.mean(prompt_lens):.2f}")
    print(f"avg. response length: {np.mean(res_lens):.2f}")

    print("\n- Histogram -")
    bin_edges = [0, 1024, 2048, 4096, 8192, 16384, 32768]
    hist = np.histogram(sample_lens, bins=bin_edges)[0]
    for i in range(len(hist)):
        print(f"L{bin_edges[i]} - {bin_edges[i+1]}: {hist[i]}")
