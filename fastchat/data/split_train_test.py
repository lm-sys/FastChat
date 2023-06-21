"""
Split the dataset into training and test set.

Usage: python3 -m fastchat.data.split_train_test --in sharegpt.json
"""
import argparse
import json

import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--ratio", type=float, default=0.9)
    args = parser.parse_args()

    content = json.load(open(args.in_file, "r"))
    np.random.seed(0)

    perm = np.random.permutation(len(content))
    content = [content[i] for i in perm]
    split = int(args.ratio * len(content))

    train_set = content[:split]
    test_set = content[split:]

    print(f"#train: {len(train_set)}, #test: {len(test_set)}")
    train_name = args.in_file.replace(".json", "_train.json")
    test_name = args.in_file.replace(".json", "_test.json")
    json.dump(train_set, open(train_name, "w"), indent=2, ensure_ascii=False)
    json.dump(test_set, open(test_name, "w"), indent=2, ensure_ascii=False)
