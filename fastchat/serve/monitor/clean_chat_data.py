"""
Clean chatbot arena battle log.

Usage:
python3 clean_battle_data.py --mode conv_release
"""
import argparse
import datetime
import json
import os
from pytz import timezone
import time

from tqdm import tqdm

from fastchat.serve.monitor.basic_stats import get_log_files, NUM_SERVERS
from fastchat.serve.monitor.clean_battle_data import to_openai_format
from fastchat.utils import detect_language


NETWORK_ERROR_MSG = (
    "NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.".lower()
)


def get_log_files(max_num_files=None):
    dates = []
    for month in [4, 5, 6, 7]:
        for day in range(1, 32):
            dates.append(f"2023-{month:02d}-{day:02d}")

    for month in [8]:
        for day in range(1, 32):
            dates.append(f"2023-{month:02d}-{day:02d}")

    filenames = []
    for d in dates:
        for i in range(NUM_SERVERS):
            name = os.path.expanduser(f"~/fastchat_logs/server{i}/{d}-conv.json")
            if os.path.exists(name):
                filenames.append(name)
    max_num_files = max_num_files or len(filenames)
    filenames = list(reversed(filenames))
    filenames = filenames[-max_num_files:]
    return filenames


def clean_chat_data(log_files):
    raw_data = []
    for filename in tqdm(log_files, desc="read files"):
        for retry in range(5):
            try:
                lines = open(filename).readlines()
                break
            except FileNotFoundError:
                time.sleep(2)

        for l in lines:
            row = json.loads(l)
            if row["type"] == "chat":
                raw_data.append(row)

    all_models = set()
    all_ips = dict()
    chats = []
    ct_invalid_conv_id = 0
    ct_invalid = 0
    ct_network_error = 0
    for row in raw_data:
        if "conv_id" not in row["state"]:
            ct_invalid_conv_id += 1
            continue

        conversation_id = row["state"]["conv_id"]
        if conversation_id is None:
            ct_invalid_conv_id += 1
            continue

        state = row["state"]
        conversation = to_openai_format(state["messages"][state["offset"] :])
        model = row["model"]
        if not isinstance(model, str):
            ct_invalid += 1
            continue

        try:
            lang_code = detect_language(state["messages"][state["offset"]][1])
        except IndexError:
            ct_invalid += 1
            continue

        if not all(isinstance(x["content"], str) for x in conversation):
            ct_invalid += 1
            continue

        messages = "".join([x["content"] for x in conversation]).lower()
        if NETWORK_ERROR_MSG in messages:
            ct_network_error += 1
            continue

        ip = row["ip"]
        if ip not in all_ips:
            all_ips[ip] = len(all_ips)
        user_id = all_ips[ip]

        chats.append(
            dict(
                conversation_id=conversation_id,
                model=model,
                conversation=conversation,
                turn=len(conversation) // 2,
                language=lang_code,
                user_id=user_id,
                tstamp=row["tstamp"],
            )
        )

        all_models.update([model])

    chats.sort(key=lambda x: x["tstamp"])
    last_updated_tstamp = chats[-1]["tstamp"]
    last_updated_datetime = datetime.datetime.fromtimestamp(
        last_updated_tstamp, tz=timezone("US/Pacific")
    ).strftime("%Y-%m-%d %H:%M:%S %Z")

    print(f"#raw: {len(raw_data)}, #chat: {len(chats)}")
    print(
        f"#invalid_conv_id: {ct_invalid_conv_id}, #network_error: {ct_network_error}, #invalid: {ct_invalid}"
    )
    print(f"#models: {len(all_models)}, {all_models}")
    print(f"last-updated: {last_updated_datetime}")

    return chats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-num-files", type=int)
    parser.add_argument(
        "--mode", type=str, choices=["simple", "conv_release"], default="simple"
    )
    args = parser.parse_args()

    log_files = get_log_files(args.max_num_files)
    chats = clean_chat_data(log_files)
    last_updated_tstamp = chats[-1]["tstamp"]
    cutoff_date = datetime.datetime.fromtimestamp(
        last_updated_tstamp, tz=timezone("US/Pacific")
    ).strftime("%Y%m%d")

    output = f"clean_chat_conv_{cutoff_date}.json"
    with open(output, "w") as fout:
        json.dump(chats, fout, indent=2, ensure_ascii=False)
    print(f"Write cleaned data to {output}")
