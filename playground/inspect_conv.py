import argparse
import code
import datetime
import json
import os
from pytz import timezone
import time

import pandas as pd
from tqdm import tqdm


def get_log_files(max_num_files=None):
    dates = []
    for month in [4, 5]:
        for day in range(1, 32):
            dates.append(f"2023-{month:02d}-{day:02d}")

    num_servers = 12
    filenames = []
    for d in dates:
        for i in range(num_servers):
            name = os.path.expanduser(f"~/fastchat_logs/server{i}/{d}-conv.json")
            if os.path.exists(name):
                filenames.append(name)
    max_num_files = max_num_files or len(filenames)
    filenames = filenames[-max_num_files:]
    return filenames


def pretty_print_conversation(messages):
    for role, msg in messages:
        print(f"[[{role}]]: {msg}")


def inspect_convs(log_files):
    data = []
    for filename in tqdm(log_files, desc="read files"):
        for retry in range(5):
            try:
                lines = open(filename).readlines()
                break
            except FileNotFoundError:
                time.sleep(2)

        for l in lines:
            row = json.loads(l)

            if "states" not in row:
                continue
            if row["type"] not in ["leftvote", "rightvote", "bothbad_vote"]:
                continue

            model_names = row["states"][0]["model_name"], row["states"][1]["model_name"]
            if row["type"] == "leftvote":
                winner, loser = model_names[0], model_names[1]
                winner_conv, loser_conv = row["states"][0], row["states"][1]
            elif row["type"] == "rightvote":
                loser, winner = model_names[0], model_names[1]
                loser_conv, winner_conv = row["states"][0], row["states"][1]

            if loser == "bard" and winner == "vicuna-13b":
                print("=" * 20)
                print(f"Winner: {winner}")
                pretty_print_conversation(winner_conv["messages"])
                print(f"Loser: {loser}")
                pretty_print_conversation(loser_conv["messages"])
                print("=" * 20)
                input()

            # if row["type"] == "bothbad_vote" and "gpt-4" in model_names:
            #    print("=" * 20)
            #    print(f"Model A: {model_names[0]}")
            #    pretty_print_conversation(row["states"][0]["messages"])
            #    print(f"Model B: {model_names[1]}")
            #    pretty_print_conversation(row["states"][1]["messages"])
            #    print("=" * 20)
            #    input()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-num-files", type=int)
    args = parser.parse_args()

    log_files = get_log_files(args.max_num_files)
    inspect_convs(log_files)
