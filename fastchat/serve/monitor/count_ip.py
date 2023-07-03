"""
Count the chat calls made by each ip address.
"""
import argparse
from collections import defaultdict
import json
import os
import time

import numpy as np
from tqdm import tqdm


def get_log_files(max_num_files=None):
    dates = []
    for month in [6]:
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


def count_ips(log_files):
    ip_counter = {}

    for filename in tqdm(log_files, desc="read files"):
        for retry in range(5):
            try:
                lines = open(filename).readlines()
                break
            except FileNotFoundError:
                time.sleep(2)

        for l in lines:
            row = json.loads(l)

            if row["type"] not in ["chat"]:
                continue

            model = row["model"]
            ip = row["ip"]

            if model not in ip_counter:
                ip_counter[model] = defaultdict(lambda: 0)

            ip_counter[model][ip] += 1

    models = ["claude-v1", "chatglm-6b", "chatglm2-6b"]
    top_k = 20

    for m in models:
        ips = list(ip_counter[m].keys())
        counts = list(ip_counter[m].values())

        indices = np.argsort(counts)[::-1]
        print(m)
        for i in indices[:top_k]:
            print(ips[i], counts[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-num-files", type=int)
    args = parser.parse_args()

    log_files = get_log_files(args.max_num_files)
    count_ips(log_files)
