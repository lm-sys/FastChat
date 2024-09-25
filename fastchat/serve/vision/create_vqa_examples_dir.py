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

"""
Creates a directory with images and JSON files for VQA examples. Final json is located in metadata_sampled.json
"""


def download_images_and_create_json(
    dataset_info, cache_dir="~/vqa_examples_cache", base_dir="./vqa_examples"
):
    for dataset_name, info in dataset_info.items():
        dataset_cache_dir = os.path.join(cache_dir, dataset_name)
        os.makedirs(dataset_cache_dir, exist_ok=True)

        if info["subset"]:
            dataset = load_dataset(
                info["path"],
                info["subset"],
                cache_dir=dataset_cache_dir,
                split=info["split"],
            )
        else:
            dataset = load_dataset(
                info["path"], cache_dir=dataset_cache_dir, split=info["split"]
            )
        dataset_dir = os.path.join(base_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        json_data = []
        for i, item in enumerate(tqdm.tqdm(dataset)):
            id_key = i if info["id_key"] == "index" else item[info["id_key"]]
            image_pil = item[info["image_key"]].convert("RGB")
            image_path = os.path.join(dataset_dir, f"{id_key}.jpg")
            image_pil.save(image_path)
            json_entry = {
                "dataset": dataset_name,
                "question": item[info["question_key"]],
                "path": image_path,
            }
            json_data.append(json_entry)

        with open(os.path.join(dataset_dir, "data.json"), "w") as json_file:
            json.dump(json_data, json_file, indent=4)
        # Delete the cache directory for the dataset
        shutil.rmtree(dataset_cache_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="~/.cache")
    parser.add_argument("--output_dir", type=str, default="./vqa_examples")
    args = parser.parse_args()

    datasets_info = {
        "Memes": {
            "path": "not-lain/meme-dataset",
            "image_key": "image",
            "question_key": "name",
            "id_key": "index",
            "subset": False,
            "split": "train",
        },
        "Floorplan": {
            "path": "umesh16071973/Floorplan_Dataset_21022024",
            "image_key": "image",
            "question_key": "caption",
            "id_key": "index",
            "subset": False,
            "split": "train",
        },
        "Website": {
            "path": "Zexanima/website_screenshots_image_dataset",
            "image_key": "image",
            "question_key": "date_captured",
            "id_key": "index",
            "subset": False,
            "split": "train",
        },
        "IllusionVQA": {
            "path": "csebuetnlp/illusionVQA-Comprehension",
            "image_key": "image",
            "question_key": "question",
            "id_key": "index",
            "subset": False,
            "split": "test",
        },
        "NewYorker": {
            "path": "jmhessel/newyorker_caption_contest",
            "image_key": "image",
            "question_key": "questions",
            "id_key": "index",
            "subset": "explanation",
            "split": "train",
        },
    }

    download_images_and_create_json(
        datasets_info, cache_dir=args.data_dir, base_dir=args.output_dir
    )
    dataset_json = []
    for dataset_name in datasets_info.keys():
        with open(f"{args.output_dir}/{dataset_name}/data.json") as f:
            data = json.load(f)
            print(f"Dataset: {dataset_name}, Number of examples: {len(data)}")
            dataset_json.extend(data)

    with open(f"{args.output_dir}/metadata_sampled.json", "w") as f:
        json.dump(dataset_json, f, indent=4)
