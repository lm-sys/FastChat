"""
Changes proportion of examples in metadata_sampled.json

Usage:

python3 -m fastchat.serve.vision.create_vqa_examples_json
"""

import json
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="~/.cache")
    parser.add_argument("--output_dir", type=str, default="./vqa_examples")
    args = parser.parse_args()

    dataset_prop = {
        "Memes": 500,
        "Floorplan": 500,
        "Website": 500,
        "IllusionVQA": 435,
        "NewYorker": 500,
    }

    dataset_json = []
    for dataset_name in dataset_prop.keys():
        with open(f"{args.output_dir}/{dataset_name}/data.json") as f:
            data = json.load(f)
            dataset_json.extend(
                np.random.choice(
                    data, min(dataset_prop[dataset_name], len(data)), replace=False
                )
            )

    with open(f"{args.output_dir}/metadata_sampled.json", "w") as f:
        json.dump(dataset_json, f, indent=4)
