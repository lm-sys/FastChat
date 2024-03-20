"""
Usage:
python3 pretty_json.py --in in.json --out out.json
"""

import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, required=True)
    args = parser.parse_args()

    with open(args.in_file, "r") as fin:
        data = json.load(fin)

    with open(args.out_file, "w") as fout:
        json.dump(data, fout, indent=2, ensure_ascii=False)
