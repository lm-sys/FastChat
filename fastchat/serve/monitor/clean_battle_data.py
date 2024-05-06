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
from multiprocessing import Pool
import tiktoken
from collections import Counter
import shortuuid

from fastchat.serve.monitor.basic_stats import get_log_files, NUM_SERVERS
from fastchat.utils import detect_language


VOTES = ["tievote", "leftvote", "rightvote", "bothbad_vote"]
IDENTITY_WORDS = [
    "vicuna",
    "lmsys",
    "koala",
    "uc berkeley",
    "open assistant",
    "laion",
    "chatglm",
    "chatgpt",
    "gpt-4",
    "openai",
    "anthropic",
    "claude",
    "bard",
    "palm",
    "lamda",
    "google",
    "gemini",
    "llama",
    "qianwan",
    "qwen",
    "alibaba",
    "mistral",
    "zhipu",
    "KEG lab",
    "01.AI",
    "AI2",
    "Tülu",
    "Tulu",
    "deepseek",
    "hermes",
    "cohere",
    "DBRX",
    "databricks",
]

ERROR_WORDS = [
    "NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.",
    "$MODERATION$ YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES.",
    "API REQUEST ERROR. Please increase the number of max tokens.",
    "**API REQUEST ERROR** Reason: The response was blocked.",
    "**API REQUEST ERROR**",
]

UNFINISHED_WORDS = [
    "▌",
    '<span class="cursor">',
]

for i in range(len(IDENTITY_WORDS)):
    IDENTITY_WORDS[i] = IDENTITY_WORDS[i].lower()

for i in range(len(ERROR_WORDS)):
    ERROR_WORDS[i] = ERROR_WORDS[i].lower()


def remove_html(raw):
    if isinstance(raw, str) and raw.startswith("<h3>"):
        return raw[raw.find(": ") + 2 : -len("</h3>\n")]
    return raw


def to_openai_format(messages):
    roles = ["user", "assistant"]
    ret = []
    for i, x in enumerate(messages):
        ret.append({"role": roles[i % 2], "content": x[1]})
    return ret


def replace_model_name(old_name, tstamp):
    replace_dict = {
        "bard": "palm-2",
        "claude-v1": "claude-1",
        "claude-instant-v1": "claude-instant-1",
        "oasst-sft-1-pythia-12b": "oasst-pythia-12b",
        "claude-2": "claude-2.0",
        "StripedHyena-Nous-7B": "stripedhyena-nous-7b",
        "gpt-4-turbo": "gpt-4-1106-preview",
        "gpt-4-0125-assistants-api": "gpt-4-turbo-browsing",
    }
    if old_name in ["gpt-4", "gpt-3.5-turbo"]:
        if tstamp > 1687849200:
            return old_name + "-0613"
        else:
            return old_name + "-0314"
    if old_name in replace_dict:
        return replace_dict[old_name]
    return old_name


def read_file(filename):
    data = []
    for retry in range(5):
        try:
            # lines = open(filename).readlines()
            for l in open(filename):
                row = json.loads(l)
                if row["type"] in VOTES:
                    data.append(row)
            break
        except FileNotFoundError:
            time.sleep(2)
    return data


def read_file_parallel(log_files, num_threads=16):
    data_all = []
    with Pool(num_threads) as p:
        ret_all = list(tqdm(p.imap(read_file, log_files), total=len(log_files)))
        for ret in ret_all:
            data_all.extend(ret)
    return data_all


def process_data(
    data,
    exclude_model_names,
    sanitize_ip,
    ban_ip_list,
):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    convert_type = {
        "leftvote": "model_a",
        "rightvote": "model_b",
        "tievote": "tie",
        "bothbad_vote": "tie (bothbad)",
    }

    all_ips = dict()

    count_dict = {
        "anony": 0,
        "invalid": 0,
        "leaked_identity": 0,
        "banned": 0,
        "error": 0,
        "unfinished": 0,
        "none_msg": 0,
        "exclude_model": 0,
    }
    count_leak = {}

    battles = []
    for row in data:
        flag_anony = False
        flag_leaked_identity = False
        flag_error = False
        flag_unfinished = False
        flag_none_msg = False

        if row["models"][0] is None or row["models"][1] is None:
            continue

        # Resolve model names
        models_public = [remove_html(row["models"][0]), remove_html(row["models"][1])]
        if "model_name" in row["states"][0]:
            models_hidden = [
                row["states"][0]["model_name"],
                row["states"][1]["model_name"],
            ]
            if models_hidden[0] is None:
                models_hidden = models_public
        else:
            models_hidden = models_public

        if (models_public[0] == "" and models_public[1] != "") or (
            models_public[1] == "" and models_public[0] != ""
        ):
            count_dict["invalid"] += 1
            continue

        if models_public[0] == "" or models_public[0] == "Model A":
            flag_anony = True
            models = models_hidden
        else:
            flag_anony = False
            models = models_public
            if (
                models_hidden[0] not in models_public[0]
                or models_hidden[1] not in models_public[1]
            ):
                count_dict["invalid"] += 1
                continue

        # Detect langauge
        state = row["states"][0]
        if state["offset"] >= len(state["messages"]):
            count_dict["invalid"] += 1
            continue
        lang_code = detect_language(state["messages"][state["offset"]][1])

        # Drop conversations if the model names are leaked
        messages = ""
        for i in range(2):
            state = row["states"][i]
            for _, (role, msg) in enumerate(state["messages"][state["offset"] :]):
                if msg:
                    messages += msg.lower()
                else:
                    flag_none_msg = True

        for word in IDENTITY_WORDS:
            if word in messages:
                if word not in count_leak:
                    count_leak[word] = 0
                count_leak[word] += 1
                flag_leaked_identity = True
                break

        for word in ERROR_WORDS:
            if word in messages:
                flag_error = True
                break

        for word in UNFINISHED_WORDS:
            if word in messages:
                flag_unfinished = True
                break

        if flag_none_msg:
            count_dict["none_msg"] += 1
            continue
        if flag_leaked_identity:
            count_dict["leaked_identity"] += 1
            continue
        if flag_error:
            count_dict["error"] += 1
            continue
        if flag_unfinished:
            count_dict["unfinished"] += 1
            continue

        # Replace bard with palm
        models = [replace_model_name(m, row["tstamp"]) for m in models]
        # Exclude certain models
        if exclude_model_names and any(x in exclude_model_names for x in models):
            count_dict["exclude_model"] += 1
            continue

        question_id = row["states"][0]["conv_id"]
        conversation_a = to_openai_format(
            row["states"][0]["messages"][row["states"][0]["offset"] :]
        )
        conversation_b = to_openai_format(
            row["states"][1]["messages"][row["states"][1]["offset"] :]
        )

        ip = row["ip"]
        if ip not in all_ips:
            all_ips[ip] = {"ip": ip, "count": 0, "sanitized_id": shortuuid.uuid()}
        all_ips[ip]["count"] += 1
        if sanitize_ip:
            user_id = f"{all_ips[ip]['sanitized_id']}"
        else:
            user_id = f"{all_ips[ip]['ip']}"

        if ban_ip_list is not None and ip in ban_ip_list:
            count_dict["banned"] += 1
            continue

        if flag_anony:
            count_dict["anony"] += 1

        for conv in conversation_a:
            conv["num_tokens"] = len(
                encoding.encode(conv["content"], allowed_special="all")
            )
        for conv in conversation_b:
            conv["num_tokens"] = len(
                encoding.encode(conv["content"], allowed_special="all")
            )

        # Save the results
        battles.append(
            dict(
                question_id=question_id,
                model_a=models[0],
                model_b=models[1],
                winner=convert_type[row["type"]],
                judge=f"arena_user_{user_id}",
                conversation_a=conversation_a,
                conversation_b=conversation_b,
                turn=len(conversation_a) // 2,
                anony=flag_anony,
                language=lang_code,
                tstamp=row["tstamp"],
            )
        )
    return battles, count_dict, count_leak, all_ips


def clean_battle_data(
    log_files,
    exclude_model_names,
    ban_ip_list=None,
    sanitize_ip=False,
    anony_only=False,
    num_threads=16,
):
    data = read_file_parallel(log_files, num_threads=16)

    battles = []
    count_dict = {}
    count_leak = {}
    all_ips = {}
    with Pool(num_threads) as p:
        # split data into chunks
        chunk_size = len(data) // min(100, len(data))
        data_chunks = [
            data[i : i + chunk_size] for i in range(0, len(data), chunk_size)
        ]

        args_list = [
            (data_chunk, exclude_model_names, sanitize_ip, ban_ip_list)
            for data_chunk in data_chunks
        ]
        ret_all = list(tqdm(p.starmap(process_data, args_list), total=len(data_chunks)))

        for ret in ret_all:
            sub_battles, sub_count_dict, sub_count_leak, sub_all_ips = ret
            battles.extend(sub_battles)
            count_dict = dict(Counter(count_dict) + Counter(sub_count_dict))
            count_leak = dict(Counter(count_leak) + Counter(sub_count_leak))
            for ip in sub_all_ips:
                if ip not in all_ips:
                    all_ips[ip] = sub_all_ips[ip]
                else:
                    all_ips[ip]["count"] += sub_all_ips[ip]["count"]
    battles.sort(key=lambda x: x["tstamp"])
    last_updated_tstamp = battles[-1]["tstamp"]

    last_updated_datetime = datetime.datetime.fromtimestamp(
        last_updated_tstamp, tz=timezone("US/Pacific")
    ).strftime("%Y-%m-%d %H:%M:%S %Z")

    print(f"#votes: {len(data)}")
    print(count_dict)
    print(f"#battles: {len(battles)}, #anony: {count_dict['anony']}")
    print(f"last-updated: {last_updated_datetime}")
    print(f"leaked_identity: {count_leak}")

    if ban_ip_list is not None:
        for ban_ip in ban_ip_list:
            if ban_ip in all_ips:
                del all_ips[ban_ip]
    print("Top 30 IPs:")
    print(sorted(all_ips.values(), key=lambda x: x["count"], reverse=True)[:30])
    return battles


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-num-files", type=int)
    parser.add_argument(
        "--mode", type=str, choices=["simple", "conv_release"], default="simple"
    )
    parser.add_argument("--exclude-model-names", type=str, nargs="+")
    parser.add_argument("--ban-ip-file", type=str)
    parser.add_argument("--sanitize-ip", action="store_true", default=False)
    args = parser.parse_args()

    log_files = get_log_files(args.max_num_files)
    ban_ip_list = json.load(open(args.ban_ip_file)) if args.ban_ip_file else None

    battles = clean_battle_data(
        log_files, args.exclude_model_names or [], ban_ip_list, args.sanitize_ip
    )
    last_updated_tstamp = battles[-1]["tstamp"]
    cutoff_date = datetime.datetime.fromtimestamp(
        last_updated_tstamp, tz=timezone("US/Pacific")
    ).strftime("%Y%m%d")

    if args.mode == "simple":
        for x in battles:
            for key in [
                "conversation_a",
                "conversation_b",
                "question_id",
            ]:
                del x[key]
        print("Samples:")
        for i in range(4):
            print(battles[i])
        output = f"clean_battle_{cutoff_date}.json"
    elif args.mode == "conv_release":
        new_battles = []
        for x in battles:
            if not x["anony"]:
                continue
            for key in []:
                del x[key]
            new_battles.append(x)
        battles = new_battles
        output = f"clean_battle_conv_{cutoff_date}.json"

    with open(output, "w", encoding="utf-8", errors="replace") as fout:
        json.dump(battles, fout, indent=2, ensure_ascii=False)
    print(f"Write cleaned data to {output}")
