"""
Usage:
python3 -m fastchat.data.inspect --in sharegpt_20230322_clean_lang_split.json
"""
import argparse
import json
import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--begin", type=int)
    parser.add_argument("--random-n", type=int)
    args = parser.parse_args()

    content = json.load(open(args.in_file, "r"))

    if args.random_n:
        indices = [random.randint(0, len(content) - 1) for _ in range(args.random_n)]
    elif args.begin:
        indices = range(args.begin, len(content))
    else:
        indices = range(0, len(content))

    for idx in indices:
        sample = content[idx]
        print("=" * 40)
        print(f"no: {idx}, id: {sample['id']}")
        for conv in sample["conversations"]:
            print(conv["from"] + ": ")
            print(conv["value"])
            input()
