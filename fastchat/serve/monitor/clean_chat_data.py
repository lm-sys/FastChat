"""
Clean chatbot arena chat log.

Usage:
python3 clean_chat_data.py
"""
import argparse
import json
import os
import hashlib
from pytz import timezone
from functools import partial
from datetime import datetime, timedelta
import time
import multiprocessing as mp

from fastchat.serve.monitor.basic_stats import NUM_SERVERS
from fastchat.serve.monitor.clean_battle_data import (
    to_openai_format,
    replace_model_name,
)
from fastchat.utils import detect_language


NETWORK_ERROR_MSG = (
    "NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.".lower()
)
MANAGER = mp.Manager()
LOCK = MANAGER.Lock()


def date_range(start="2023-04-01"):
    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.now().date()
    delta = end_date - start_date
    dates = [
        (start_date + timedelta(days=d)).strftime("%Y-%m-%d")
        for d in range(delta.days + 2)
    ]

    return dates


def get_log_files(max_num_files=None):
    dates = date_range()
    filenames = []
    for d in dates:
        for i in range(NUM_SERVERS):
            name = os.path.expanduser(f"~/fastchat_logs/server{i}/{d}-conv.json")
            if os.path.exists(name):
                filenames.append(name)
    max_num_files = max_num_files or len(filenames)
    # filenames = list(reversed(filenames))
    filenames = filenames[-max_num_files:]
    return filenames


def get_action_type_data(filename, action_type):
    for _ in range(5):
        try:
            lines = open(filename).readlines()
            break
        except FileNotFoundError:
            time.sleep(2)

    rows = []
    for l in lines:
        row = json.loads(l)
        if row["type"] == action_type:
            rows.append(row)
    return rows


def process_data(row, action_type):
    try:
        if action_type in ["chat", "upvote", "downvote"]:
            state = row["state"]
            model = row["model"]
        elif action_type == "leftvote":
            state = row["states"][0]
            model = row["states"][0]["model_name"]
        elif action_type == "rightvote":
            state = row["states"][1]
            model = row["states"][1]["model_name"]
        conversation_id = state["conv_id"]
    except KeyError:
        return {
            "result": None,
            "ct_invalid_conv_id": 1,
            "ct_invalid": 0,
            "ct_network_error": 0,
            "model": None,
        }

    if conversation_id is None:
        return {
            "result": None,
            "ct_invalid_conv_id": 1,
            "ct_invalid": 0,
            "ct_network_error": 0,
            "model": None,
        }

    conversation = to_openai_format(state["messages"][state["offset"] :])
    if not isinstance(model, str):
        return {
            "result": None,
            "ct_invalid_conv_id": 0,
            "ct_invalid": 1,
            "ct_network_error": 0,
            "model": None,
        }
    model = replace_model_name(model, row["tstamp"])

    try:
        lang_code = detect_language(state["messages"][state["offset"]][1])
    except IndexError:
        return {
            "result": None,
            "ct_invalid_conv_id": 0,
            "ct_invalid": 1,
            "ct_network_error": 0,
            "model": None,
        }

    if not all(isinstance(x["content"], str) for x in conversation):
        return {
            "result": None,
            "ct_invalid_conv_id": 0,
            "ct_invalid": 1,
            "ct_network_error": 0,
            "model": None,
        }

    messages = "".join([x["content"] for x in conversation]).lower()
    if NETWORK_ERROR_MSG in messages:
        return {
            "result": None,
            "ct_invalid_conv_id": 0,
            "ct_invalid": 0,
            "ct_network_error": 1,
            "model": None,
        }
    user_id = hashlib.md5(row["ip"].encode()).hexdigest()

    # Prepare the result data
    result = dict(
        conversation_id=conversation_id,
        model=model,
        conversation=conversation,
        turn=len(conversation) // 2,
        language=lang_code,
        user_id=user_id,
        tstamp=row["tstamp"],
    )

    return {
        "result": result,
        "ct_invalid_conv_id": 0,
        "ct_invalid": 0,
        "ct_network_error": 0,
        "model": model,
    }


def clean_chat_data(log_files, action_type):
    with mp.Pool() as pool:
        # Use partial to pass action_type to get_action_type_data
        func = partial(get_action_type_data, action_type=action_type)
        file_data = pool.map(func, log_files, chunksize=1)
    # filter out Nones as some files may not contain any data belong to action_type
    raw_data = []
    for data in file_data:
        raw_data.extend(data)
    raw_data = [r for r in raw_data if r is not None]

    # Use the multiprocessing Pool
    with mp.Pool() as pool:
        func = partial(process_data, action_type=action_type)
        results = pool.map(func, raw_data, chunksize=1)

    # Aggregate results from child processes
    ct_invalid_conv_id = sum([data["ct_invalid_conv_id"] for data in results])
    ct_invalid = sum([data["ct_invalid"] for data in results])
    ct_network_error = sum([data["ct_network_error"] for data in results])
    all_models = set([data["model"] for data in results if not (data["model"] is None)])
    chats = [data["result"] for data in results if not (data["result"] is None)]

    chats.sort(key=lambda x: x["tstamp"])
    last_updated_tstamp = chats[-1]["tstamp"]
    last_updated_datetime = datetime.fromtimestamp(
        last_updated_tstamp, tz=timezone("US/Pacific")
    ).strftime("%Y-%m-%d %H:%M:%S %Z")

    # Deduplication
    dedup_chats = []
    visited_conv_ids = set()
    for i in reversed(range(len(chats))):
        if chats[i]["conversation_id"] in visited_conv_ids:
            continue
        visited_conv_ids.add(chats[i]["conversation_id"])
        dedup_chats.append(chats[i])

    print(
        f"#raw: {len(raw_data)}, #chat: {len(chats)}, #dedup_chat: {len(dedup_chats)}"
    )
    print(
        f"#invalid_conv_id: {ct_invalid_conv_id}, #network_error: {ct_network_error}, #invalid: {ct_invalid}"
    )
    print(f"#models: {len(all_models)}, {all_models}")
    print(f"last-updated: {last_updated_datetime}")

    return list(reversed(dedup_chats))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-type", type=str, default="chat")
    parser.add_argument("--max-num-files", type=int)
    args = parser.parse_args()

    log_files = get_log_files(args.max_num_files)
    chats = clean_chat_data(log_files, args.action_type)
    last_updated_tstamp = chats[-1]["tstamp"]
    cutoff_date = datetime.fromtimestamp(
        last_updated_tstamp, tz=timezone("US/Pacific")
    ).strftime("%Y%m%d")

    output = f"clean_{args.action_type}_conv_{cutoff_date}.json"
    with open(output, "w") as fout:
        json.dump(chats, fout, indent=2, ensure_ascii=False)
    print(f"Write cleaned data to {output}")
