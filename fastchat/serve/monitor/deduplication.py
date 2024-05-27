import os
import json
import pandas as pd
import ast

import matplotlib.pyplot as plt
from matplotlib import rcParams

import argparse
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--percentile", type=float, default=0.9999)
    args = parser.parse_args()
    output_dir = args.output_dir
    input_file = args.input_file

    with open(input_file) as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Preprocessing
    all_convs_new = []
    convs = []
    for row in data:
        conv = ""
        for turns in row["conversation_a"]:
            if turns["role"] == "user":
                conv += f"{turns['content']}\n"

        convs.append(conv[:10000])
        row["post_process_conv"] = conv[:10000]
        all_convs_new.append(row)

    df = pd.DataFrame(all_convs_new)
    print("Number of conversations: ", len(df))

    prompt_counts = df["post_process_conv"].value_counts()
    # Select the top 20 most frequent prompts
    top_prompts = prompt_counts.head(20)
    print(top_prompts)

    # Determine the percentile count
    percentile_cutoff = prompt_counts.quantile(args.percentile)
    print(f"{args.percentile*100} percentile count: {percentile_cutoff}")

    # prompts that are more common than the percentile cutoff
    high_frequency_prompts = prompt_counts[prompt_counts > percentile_cutoff].index
    print(
        f"Number of high frequency prompts: {len(high_frequency_prompts)}/{len(prompt_counts)}"
    )

    # initialize a new column dedup_tag
    dedup_tags = np.array(
        [{"high_freq": False, "sampled": True} for _ in range(len(df))]
    )
    high_freq_groups = df.groupby("post_process_conv")
    for prompt in tqdm(high_frequency_prompts):
        df_high_freq = high_freq_groups.get_group(prompt)
        sampled_indices = df_high_freq.sample(
            n=int(percentile_cutoff), random_state=42
        ).index
        dedup_tags[df_high_freq.index] = {"high_freq": True, "sampled": False}
        dedup_tags[sampled_indices] = {"high_freq": True, "sampled": True}

    df["dedup_tag"] = dedup_tags

    # drop intermediate columns (post_process_conv)
    df = df.drop(columns=["post_process_conv"])

    df.to_json(
        os.path.join(output_dir, "dedup.json"),
        orient="records",
        indent=4,
        force_ascii=False,
    )
