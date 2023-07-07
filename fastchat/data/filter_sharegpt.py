"""
Filter some invalid samples from ShareGPT dataset from anon8231489123/ShareGPT_Vicuna_unfiltered

Usage:
python3 -m fastchat.data.filter_sharegpt --in ShareGPT_V3_unfiltered_cleaned_split.json --out sharegpt_filtered.json
"""
import argparse
import json
from fastchat.model.model_adapter import get_conversation_template


def main(args):
    with open(args["in_file"], "r") as in_file:
        raw_data = json.load(in_file)

    conv = get_conversation_template(args["template"])
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    filtered_data = []
    for i, example in enumerate(raw_data):
        source = example["conversations"]
        try:
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]
            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            filtered_data.append(example)  # Add the example to filtered data if no error was raised
        except:
            continue  # skip invalid conversations

    # Save the valid conversations
    with open(args["out_file"], "w") as out_file:
        json.dump(filtered_data, out_file, indent=2)
    
    print(f"Filtered {len(raw_data) - len(filtered_data)} invalid conversations from {len(raw_data)} total conversations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, default="sharegpt_filtered.json")
    parser.add_argument("--template", type=str, default="vicuna")
    args = parser.parse_args()
    main(vars(args))