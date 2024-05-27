import datasets
from datasets import load_dataset
from PIL import Image
from pathlib import Path
import pandas as pd
import os
import json
import tqdm
import argparse
import shutil
import numpy as np

np.random.seed(0)


def download_images_and_create_json(
    dataset_info, cache_dir="~/vqa_examples_cache", base_dir="./vqa_examples"
):
    for dataset_name, info in dataset_info.items():
        dataset_cache_dir = os.path.join(cache_dir, dataset_name)
        os.makedirs(dataset_cache_dir, exist_ok=True)

        if info["subset"]:
            dataset = load_dataset(
                info["path"], info["subset"], cache_dir=dataset_cache_dir, split="test"
            )
        else:
            dataset = load_dataset(
                info["path"], cache_dir=dataset_cache_dir, split="test"
            )
        dataset_dir = os.path.join(base_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        json_data = []
        # add tqdm to show progress bar
        for i, item in enumerate(tqdm.tqdm(dataset)):
            id_key = i if info["id_key"] == "index" else item[info["id_key"]]
            image_pil = item[info["image_key"]].convert("RGB")
            image_path = os.path.join(dataset_dir, f"{id_key}.jpg")
            # save the image
            image_pil.save(image_path)
            # Append data to JSON list
            json_entry = {
                "dataset": dataset_name,
                "question": item[info["question_key"]],
                "path": image_path,
            }
            json_data.append(json_entry)

        # Save the JSON data to a file
        with open(os.path.join(dataset_dir, "data.json"), "w") as json_file:
            json.dump(json_data, json_file, indent=4)
        # Delete the cache directory for the dataset
        shutil.rmtree(dataset_cache_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="~/.cache")
    parser.add_argument("--output_dir", type=str, default="./vqa_examples")
    args = parser.parse_args()

    # Define the dataset information
    datasets_info = {
        "DocVQA": {
            "path": "lmms-lab/DocVQA",
            "image_key": "image",
            "question_key": "question",
            "id_key": "questionId",
            "subset": "DocVQA",
        },
        "ChartQA": {
            "path": "HuggingFaceM4/ChartQA",
            "image_key": "image",
            "question_key": "query",
            "id_key": "index",
            "subset": False,
        },
        "realworldqa": {
            "path": "visheratin/realworldqa",
            "image_key": "image",
            "question_key": "question",
            "id_key": "index",
            "subset": False,
        },
        "VizWiz-VQA": {
            "path": "lmms-lab/VizWiz-VQA",
            "image_key": "image",
            "question_key": "question",
            "id_key": "question_id",
            "subset": False,
        },
    }

    download_images_and_create_json(
        datasets_info, cache_dir=args.data_dir, base_dir=args.output_dir
    )
    dataset_json = []
    for dataset_name in datasets_info.keys():
        with open(f"{args.output_dir}/{dataset_name}/data.json") as f:
            data = json.load(f)
            dataset_json.extend(np.random.choice(data, 765))

    # save dataset_json to ../vqa_examples/metadata.json
    with open(f"{args.output_dir}/metadata_sampled.json", "w") as f:
        json.dump(dataset_json, f, indent=4)
