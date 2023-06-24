"""
Filter conversations with wrong formats.

Usage:
python3 -m fastchat.data.filter_wrong_format --in input.json --out output.json

"""
import argparse
import json
import re

from tqdm import tqdm

wrong_indices_pattern = re.compile("\n1\. [^2]*\n1\. ")


def should_skip(conv):
    # Filter wrong list indices like https://sharegpt.com/c/1pREAGO
    for sentence in conv["conversations"]:
        val = sentence["value"]
        sub = re.search(wrong_indices_pattern, val)
        if sub is not None:
            return True

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, required=True)
    args = parser.parse_args()

    content = json.load(open(args.in_file, "r"))

    new_content = []
    for conv in tqdm(content):
        if should_skip(conv):
            print(f"{conv['id']} contains a wrong format.")
        else:
            new_content.append(conv)

    print(f"#in: {len(content)}, #out: {len(new_content)}")
    json.dump(new_content, open(args.out_file, "w"), indent=2, ensure_ascii=False)
