"""
Usage: python3 -m chatserver.data.clean_lang --lang en --in sharegpt_clean.json --out sharegpt_clean_en.json
"""
import argparse
import json
from langdetect import detect
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, default="")
    parser.add_argument("--lang", type=str, default="en")
    args = parser.parse_args()

    in_file = args.in_file
    out_file = args.out_file
    lang = args.lang
    if out_file == "":
        out_file = "sharegpt_clean_" + lang + ".json"
 
    content = json.load(open(in_file, "r"))
    num_conv = len(content)

    new_content = []
    for conv in tqdm(content):
        res = "NA"
        for sentence in conv["conversations"][:10]:
            try:
                res = detect(sentence["value"])
                break
            except:
                pass
        if res == lang:
            new_content.append(conv)

    print(f"return {len(new_content)} out of {len(content)}, start dump ...")
    json.dump(new_content, open(out_file, "w"), indent=2)

