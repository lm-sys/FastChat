"""
Merge two conversation files into one

Usage: python3 -m fastchat.data.merge --in file1.json file2.json --out merged.json
"""

import argparse
import json
from typing import Dict, Sequence, Optional


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True, nargs="+")
    parser.add_argument("--out-file", type=str, default="merged.json")
    args = parser.parse_args()

    new_content = []
    for in_file in args.in_file:
        content = json.load(open(in_file, "r"))
        new_content.extend(content)

    print(f"#out: {len(new_content)}")
    json.dump(new_content, open(args.out_file, "w"), indent=2, ensure_ascii=False)
