import argparse
import json

from tqdm import tqdm
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--number", type=int, default=1000000)
    args = parser.parse_args()

    # Read conversations
    convs = json.load(open(args.in_file))
    print(f"#conv: {len(convs)}")

    for c in convs:
        del c["tstamp"]
        del c["user_id"]

    np.random.seed(42)
    np.random.shuffle(convs)

    convs = convs[:args.number]
    print(f"#out conv: {len(convs)}")

    out_file = args.in_file.replace(".json", ".s2.json")
    print(f"Output to {out_file}")
    with open(out_file, "w") as fout:
        json.dump(convs, fout, indent=2, ensure_ascii=False)
