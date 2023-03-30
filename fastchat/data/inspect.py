"""
Usage:
python3 -m fastchat.data.inspect --in sharegpt_20230322_clean_lang_split.json
"""
import argparse
import json

import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--begin", type=int)
    args = parser.parse_args()

    content = json.load(open(args.in_file, "r"))
    for sample in tqdm.tqdm(content[args.begin:]):
        print(f"id: {sample['id']}")
        for conv in sample["conversations"]:
            print(conv["from"] + ": ")
            print(conv["value"])
            input()
