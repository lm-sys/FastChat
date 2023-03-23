"""
Split the long conversation based on certain max length
Usage: python3 -m chatserver.data.split_long_conversation \
    --in-file sharegpt_clean.json \
    --model-name-or-path $<model-name> \
    --out-file sharegpt_split.json \
    --max-length 1024
"""
import argparse
import json
from typing import Dict, Sequence

import transformers

DEFAULT_PAD_TOKEN = "[PAD]"
BEGIN_SIGNAL = "### "
END_SIGNAL = "\n"


def _add_speaker_and_signal(source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    conversation = ""
    for sentence in source:
        sentence["value"] = (BEGIN_SIGNAL + sentence["from"] + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    return conversation


def split_contents(raw: Sequence[Sequence[Dict]],
                   tokenizer: transformers.PreTrainedTokenizer, max_length):
    """
    Version 1 of data augmentation. Keep the maximum round of conversations
    within the max token length constraint. Not use the rest length to get more
    context.
    """
    split = []

    def split_example(example, start_idx, end_idx):
        # only ends in the bot because otherwise the last human part is useless.
        end_speaker = example["conversations"][end_idx]["from"]
        end_idx = end_idx + 1 if end_speaker != "human" else end_idx
        return {
            "id": example["id"] + "_" + str(start_idx),
            "conversations": example["conversations"][start_idx:end_idx]
        }

    # modify each sentence first.
    sources = [example["conversations"] for example in raw]
    for source in sources:
        _add_speaker_and_signal(source, get_conversation=False)

    for example in raw:
        source = example["conversations"]
        tokenized_len = [
            tokenizer(
                s["value"],
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ).input_ids.ne(tokenizer.pad_token_id).sum().item() for s in source
        ]
        num_tokens = 0
        start_idx = 0
        for idx, l in enumerate(tokenized_len):
            # TODO: shall we also only starts from a specific speaker?
            if num_tokens + l > max_length:
                split.append(split_example(example, start_idx, idx))
                start_idx = idx
                num_tokens = l
            else:
                num_tokens += l
                if idx == len(tokenized_len) - 1:
                    # already the last part.
                    split.append(split_example(example, start_idx, idx))
    # back to the previous format
    for example in split:
        source = example["conversations"]
        for s in source:
            begin_idx = len(BEGIN_SIGNAL) + len(s["from"]) + len(": ")
            s["value"] = s["value"][begin_idx:-len(END_SIGNAL)]
    return split


def main(args):
    content = json.load(open(args['in_file'], "r"))
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args["model_name_or_path"],
        model_max_length=args["max_length"],
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))
    content = split_contents(content, tokenizer, args["max_length"])
    json.dump(content, open(args['out_file'], "w"), indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--out-file", type=str, default="sharegpt_clean.json")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    main(vars(args))
