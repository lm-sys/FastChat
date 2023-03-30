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

DEFAULT_PAD_TOKEN = "[PAD]"
BEGIN_SIGNAL = "### "
END_SIGNAL = "\n"


def split_sample(sample, start_idx, end_idx):
    # only ends in the bot because otherwise the last human part is useless.
    end_speaker = sample["conversations"][end_idx]["from"]
    end_idx = end_idx + 1 if end_speaker != "human" else end_idx
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

        for c in sample["conversations"]:
            from_str = c["from"]
            if from_str.lower() == "human":
                from_str = conversation_lib.default_conversation.roles[0]
            elif from_str.lower() == "gpt":
                from_str = conversation_lib.default_conversation.roles[1]
            else:
                from_str = 'unknown'

            sentence = (BEGIN_SIGNAL + from_str + ": " + c["value"] +
                        END_SIGNAL)
            length = tokenizer(sentence, return_tensors="pt", padding="longest"
                ).input_ids.ne(tokenizer.pad_token_id).sum().item()
            tokenized_lens.append(length)

        num_tokens = 0
        start_idx = 0
        for idx, l in enumerate(tokenized_lens):
            # TODO: shall we also only starts from a specific speaker?
            if num_tokens + l > max_length:
                new_content.append(split_sample(sample, start_idx, idx))
                start_idx = idx
                num_tokens = l
            else:
                num_tokens += l
                if idx == len(tokenized_lens) - 1:
                    new_content.append(split_sample(sample, start_idx, idx))

    print(f"total: {len(content)}, new: {len(new_content)}")
    return new_content


def main(args):
    content = json.load(open(args.in_file, "r"))
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))
    content = split_contents(content, args.begin, args.end,
        tokenizer, args.max_length)
    json.dump(content, open(args.out_file, "w"), indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, default="sharegpt_split.json")
    parser.add_argument("--begin", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=2304)
    args = parser.parse_args()
    main(args)
