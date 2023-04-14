"""
Split long conversations based on certain max length.

Usage: python3 -m fastchat.data.split_long_conversation \
    --in sharegpt_clean.json \
    --out sharegpt_split.json \
    --model-name-or-path $<model-name>
"""
import argparse
import json
from typing import Dict, Sequence, Optional

import transformers
import tqdm

from fastchat import conversation as conversation_lib


def split_sample(sample, start_idx, end_idx):
    assert (end_idx - start_idx) % 2 == 0
    return {
        "id": sample["id"] + "_" + str(start_idx),
        "conversations": sample["conversations"][start_idx:end_idx]
    }


def split_contents(content, begin, end, tokenizer, max_length):
    """
    Keep the maximum round of conversations within the max token length constraint
    """
    content = content[begin:end]
    new_content = []

    for sample in tqdm.tqdm(content):
        tokenized_lens = []
        conversations = sample["conversations"]
        conversations = conversations[:len(conversations) // 2 * 2]
        for c in conversations:
            length = len(tokenizer(c["value"]).input_ids) + 5
            tokenized_lens.append(length)

        start_idx = 0
        cur_len = 0
        sample
        assert len(conversations) % 2 == 0, f"id: {sample['id']}"
        for i in range(0, len(conversations), 2):
            tmp_len = tokenized_lens[i] + tokenized_lens[i+1]
            if cur_len + tmp_len > max_length:
                new_content.append(split_sample(sample, start_idx, i))
                start_idx = i
                cur_len = 0
            elif i == len(conversations) - 2:
                new_content.append(split_sample(sample, start_idx, i+2))

            cur_len += tmp_len

    return new_content


def filter_invalid_roles(content):
    new_content = []
    for i, c in enumerate(content):
        roles = ["human", "gpt"]
        if len(c["conversations"]) <= 0:
            continue

        valid = True
        for j, s in enumerate(c["conversations"]):
            if s["from"] != roles[j % 2]:
                valid = False
                break

        if valid:
            new_content.append(c)

    return new_content


def main(args):
    content = json.load(open(args.in_file, "r"))
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=False,
    )
    new_content = split_contents(content, args.begin, args.end,
        tokenizer, args.max_length)
    new_content = filter_invalid_roles(new_content)

    print(f"total: {len(content)}, new: {len(new_content)}")
    json.dump(new_content, open(args.out_file, "w"), indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, default="sharegpt_split.json")
    parser.add_argument("--begin", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    main(args)
