"""
Usage:
python3 -m fastchat.data.optional_clean --lang en --reduce-rep --in sharegpt_clean.json --out output.json
python3 -m fastchat.data.optional_clean --skip-lang en --reduce-rep --in sharegpt_clean.json --out output.json
"""
import argparse
import json
import re

import polyglot
from polyglot.detect import Detector
import pycld2
from tqdm import tqdm


def skip(conv, args):
    # Remove certain languages
    if args.lang != "all" or args.skip_lang is not None:
        text = "\n".join([x["value"] for x in conv["conversations"]])
        try:
            lang_code = Detector(text).language.code
        except (pycld2.error, polyglot.detect.base.UnknownLanguage):
            lang_code = "unknown"

        if args.lang != "all" and lang_code != args.lang:
            return True

        if lang_code == args.skip_lang:
            return True

    # Remove repetitive numbers
    if args.reduce_rep:
        for sentence in conv["conversations"]:
            val = sentence["value"]
            sub = re.search(r"(\d)\1{8}", val)
            if sub is not None:
                return True

    return False
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, default="")
    parser.add_argument("--lang", type=str, default="all",
                        choices=["all", "en"])
    parser.add_argument("--skip-lang", type=str)
    # NOTE: Be careful about reduce_rep which may remove some good data.
    # For example, addresses could have long consecutive 0's
    parser.add_argument("--reduce-rep", action="store_true")
    args = parser.parse_args()

    in_file = args.in_file
    out_file = args.out_file
    lang = args.lang
    skip_lang = args.skip_lang
    reduce_rep = args.reduce_rep
    assert (lang == "all" or skip_lang is None)

    if out_file == "":
        out_file = "sharegpt_clean"
        if lang != "all":
            out_file += "_" + lang
        if skip_lang is not None:
            out_file += "_skip_" + skip_lang
        if reduce_rep:
            out_file += "_reduce_rep"
        out_file += ".json"
 
    content = json.load(open(in_file, "r"))
    num_conv = len(content)

    new_content = []
    for conv in tqdm(content):
        if not skip(conv, args):
            new_content.append(conv)

    print(f"return {len(new_content)} out of {len(content)}, start dump ...")
    json.dump(new_content, open(out_file, "w"), indent=2)
