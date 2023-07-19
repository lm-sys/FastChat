"""Prepare all datasets."""

import argparse
import os

from fastchat.utils import run_cmd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="~/datasets/sharegpt_20230521")
    parser.add_argument(
        "--model-name-or-path", type=str, default="meta-llama/Llama-2-7b-chat-hf"
    )
    parser.add_argument("--seq-len", type=int, default=4096)
    args = parser.parse_args()

    in_prefix = args.prefix
    model_path = args.model_name_or_path
    seq_len = args.seq_len
    prefix = (
        f"{in_prefix}_{seq_len}".replace("4096", "4k")
        .replace("8192", "8k")
        .replace("16384", "16k")
    )

    cmd_list = [
        f"python3 -m fastchat.data.clean_sharegpt --in {in_prefix}_html.json --out {prefix}_clean.json",
        f"python3 -m fastchat.data.optional_clean --in {prefix}_clean.json --out {prefix}_clean_lang.json --skip-lang ko",
        f"python3 -m fastchat.data.split_long_conversation --in {prefix}_clean_lang.json --out {prefix}_clean_lang_split.json --model-name {model_path} --max-length {seq_len}",
        f"python3 -m fastchat.data.filter_wrong_format --in {prefix}_clean_lang_split.json --out {prefix}_clean_lang_split.json",
        f"python3 -m fastchat.data.split_train_test --in {prefix}_clean_lang_split.json --ratio 0.99",
        f"python3 -m fastchat.data.hardcoded_questions",
        f"python3 -m fastchat.data.merge --in {prefix}_clean_lang_split_train.json hardcoded.json --out {prefix}_clean_lang_split_identity.json",
        f"python3 -m fastchat.data.extract_gpt4_only --in {prefix}_clean_lang_split_identity.json",
        f"python3 -m fastchat.data.extract_single_round --in {prefix}_clean_lang_split_identity.json",
    ]

    for cmd in cmd_list:
        ret = run_cmd(cmd)
        if ret != 0:
            exit(ret)
