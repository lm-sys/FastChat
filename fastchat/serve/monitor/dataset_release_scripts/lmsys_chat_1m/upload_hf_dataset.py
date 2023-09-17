"""
Upload to huggingface.
"""
import argparse
import json
from datasets import Dataset, DatasetDict, load_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    args = parser.parse_args()

    objs = json.load(open(args.in_file))
    print(f"#convs: {len(objs)}")
    data = Dataset.from_list(objs)
    data.push_to_hub("lmsys/lmsys-chat-1m", private=True)
