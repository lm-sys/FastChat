"""
Usage:
python3 replace_model_name.py --in clean_conv_20230809_10k.json
"""

import argparse
import json

from fastchat.serve.monitor.clean_battle_data import replace_model_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    args = parser.parse_args()

    convs = json.load(open(args.in_file))
    for x in convs:
        x["model"] = replace_model_name(x["model"])

    with open(args.in_file, "w") as fout:
        json.dump(convs, fout, indent=2, ensure_ascii=False)
