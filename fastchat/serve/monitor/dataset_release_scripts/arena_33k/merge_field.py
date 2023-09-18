"""Count the unique users in a battle log file."""

import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--tag-file", type=str)
    args = parser.parse_args()

    # build index
    objs = json.load(open(args.tag_file))
    new_field_dict = {}
    for obj in objs:
        new_field_dict[obj["question_id"]] = obj["toxic_chat"]

    objs = json.load(open(args.input))
    for obj in objs:
        obj["toxic_chat_tag"] = new_field_dict[obj["question_id"]]

    output = args.input.replace(".json", "_added.json")
    with open(output, "w") as fout:
        json.dump(objs, fout, indent=2, ensure_ascii=False)
