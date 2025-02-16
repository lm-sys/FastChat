"""
Extract the first round of the conversations.

Usage: python3 -m fastchat.data.extract_single_round --in sharegpt.json
"""
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str)
    parser.add_argument("--begin", type=int)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()

    content = json.load(open(args.in_file, "r"))
    content = content[args.begin : args.end]
    for c in content:
        c["conversations"] = c["conversations"][:2]

    if args.out_file:
        out_file = args.out_file
    else:
        out_file = args.in_file.replace(".json", "_single.json")

    print(f"#in: {len(content)}, #out: {len(content)}")
    json.dump(content, open(out_file, "w"), indent=2, ensure_ascii=False)
