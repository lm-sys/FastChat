"""
Count the unique users in a battle log file.

Usage:
python3 -input in.json --number 1000
"""

import argparse
import json
import random

K = 1000

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--number", type=int, nargs="+")
    args = parser.parse_args()

    convs = json.load(open(args.input))
    random.seed(42)
    random.shuffle(convs)

    for number in args.number:
        new_convs = convs[:number]

        output = args.input.replace(".json", f"_{number//K}k.json")
        with open(output, "w") as fout:
            json.dump(new_convs, fout, indent=2, ensure_ascii=False)

        print(f"#in: {len(convs)}, #out: {len(new_convs)}")
        print(f"Write to file: {output}")
