"""
Take the intersection of two conversation files.

Usage: python3 -m fastchat.data.merge --input input.json --conv-id conv_id_file.json --out intersect.json
"""

import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--conv-id", type=str, required=True)
    parser.add_argument("--out-file", type=str, default="intersect.json")
    args = parser.parse_args()

    conv_id_objs = json.load(open(args.conv_id, "r"))
    conv_ids = set(x["conversation_id"] for x in conv_id_objs)

    objs = json.load(open(args.input, "r"))
    after_objs = [x for x in objs if x["conversation_id"] in conv_ids]

    print(f"#in: {len(objs)}, #out: {len(after_objs)}")
    json.dump(after_objs, open(args.out_file, "w"), indent=2, ensure_ascii=False)
