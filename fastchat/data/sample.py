"""
Sample some conversations from a file.

Usage: python3 -m fastchat.data.sample --in sharegpt.json --out sampled.json
"""
import argparse
import json

import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, default="sampled.json")
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--keep-order", action="store_true")
    args = parser.parse_args()

    content = json.load(open(args.in_file, "r"))
    if not args.keep_order:
        np.random.seed(42)
        np.random.shuffle(content)

    new_content = []
    for i in range(args.begin, min(args.end, len(content))):
        sample = content[i]
        concat = ""
        for s in sample["conversations"]:
            concat += s["value"]

        if len(concat) > args.max_length:
            continue

        new_content.append(sample)

    print(f"#in: {len(content)}, #out: {len(new_content)}")
    json.dump(new_content, open(args.out_file, "w"), indent=2, ensure_ascii=False)
