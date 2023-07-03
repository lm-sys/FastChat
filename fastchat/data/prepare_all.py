"""Prepare all datasets."""

import os


def run_cmd(cmd):
    print(cmd, flush=True)
    return os.system(cmd)


prefix = "~/datasets/sharegpt_20230521"
llama_weights = "~/model_weights/llama-7b/"

cmd_list = [
    f"python3 -m fastchat.data.clean_sharegpt --in {prefix}_html.json --out {prefix}_clean.json",
    f"python3 -m fastchat.data.optional_clean --in {prefix}_clean.json --out {prefix}_clean_lang.json --skip-lang ko",
    f"python3 -m fastchat.data.split_long_conversation --in {prefix}_clean_lang.json --out {prefix}_clean_lang_split.json --model-name {llama_weights}",
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
