"""
Usage: python3 -m chatserver.data.optional_clean --lang en --reduce-rep --in sharegpt_clean.json --out output.json
"""
import argparse
import json
from langdetect import detect
import re
from tqdm import tqdm


def skip(conv, options: dict):
    if options["lang"] != "all":
        res = 0
        for sentence in conv["conversations"][:10]:
            try:
                lang = detect(sentence["value"])
                if lang != options["lang"]:
                    res += 1
                break
            except:
                res += 1
                pass
        if res > min(5, len(conv["conversations"]) // 2):
            return True
    if options["reduce_rep"]:
        for sentence in conv["conversations"]:
            val = sentence["value"]
            sub = re.search(r"(\d)\1{8}", val)
            if sub is not None:
                print(val)
                return True
    return False
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, default="")
    # NOTICE This will also remove data that use two languages.
    parser.add_argument("--lang", type=str, default="all",
                        choices=["all", "en"])
    # DANGER Be careful about turn on the reduce_rep, which may remove some good data.
    # For example, addresses could have long consecutive 0's
    parser.add_argument("--reduce-rep", action="store_true")
    args = parser.parse_args()

    in_file = args.in_file
    out_file = args.out_file
    lang = args.lang
    reduce_rep = args.reduce_rep

    if out_file == "":
        out_file = "sharegpt_clean"
        if lang != "all":
            out_file += "_" + lang
        if reduce_rep:
            out_file += "_reduce_rep"
        out_file += ".json"
 
    content = json.load(open(in_file, "r"))
    num_conv = len(content)

    new_content = []
    for conv in tqdm(content):
        if not skip(conv, vars(args)):
            new_content.append(conv)

    print(f"return {len(new_content)} out of {len(content)}, start dump ...")
    json.dump(new_content, open(out_file, "w"), indent=2)

