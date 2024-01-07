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
    "llama",
    "qianwan",
    "alibaba",
    "mistral",
    "zhipu",
    "KEG lab",
    "01.AI",
    "AI2",
    "Tülu",
    "Tulu",
    "NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.",
    "$MODERATION$ YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES.",
    "API REQUEST ERROR. Please increase the number of max tokens.",
    "**API REQUEST ERROR** Reason: The response was blocked.",
    "**API REQUEST ERROR**",
]

for i in range(len(IDENTITY_WORDS)):
    IDENTITY_WORDS[i] = IDENTITY_WORDS[i].lower()


def remove_html(raw):
    if raw.startswith("<h3>"):
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
    from multiprocessing import Pool

    with Pool(num_threads) as p:
        ret_all = list(tqdm(p.imap(read_file, log_files), total=len(log_files)))
        for ret in ret_all:
            data_all.extend(ret)
    return data_all


def clean_battle_data(
    log_files, exclude_model_names, ban_ip_list=None, sanitize_ip=False
):
    data = read_file_parallel(log_files, num_threads=16)

    convert_type = {
        "leftvote": "model_a",
        "rightvote": "model_b",
        "tievote": "tie",
        "bothbad_vote": "tie (bothbad)",
    }

    all_models = set()
    all_ips = dict()
    ct_anony = 0
    ct_invalid = 0
    ct_leaked_identity = 0
    ct_banned = 0
    battles = []
    for row in data:
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
            ct_invalid += 1
            continue

        if models_public[0] == "" or models_public[0] == "Model A":
            anony = True
            models = models_hidden
            ct_anony += 1
        else:
            anony = False
            models = models_public
            if not models_public == models_hidden:
                ct_invalid += 1
                continue

        # Detect langauge
        state = row["states"][0]
        if state["offset"] >= len(state["messages"]):
            ct_invalid += 1
            continue
        lang_code = detect_language(state["messages"][state["offset"]][1])

        # Drop conversations if the model names are leaked
        leaked_identity = False
        messages = ""
        for i in range(2):
            state = row["states"][i]
            for turn_idx, (role, msg) in enumerate(
                state["messages"][state["offset"] :]
            ):
                if msg:
                    messages += msg.lower()
        for word in IDENTITY_WORDS:
            if word in messages:
                leaked_identity = True
                break

        if leaked_identity:
            ct_leaked_identity += 1
            continue

        # Replace bard with palm
        models = [replace_model_name(m, row["tstamp"]) for m in models]
        # Exclude certain models
        if exclude_model_names and any(x in exclude_model_names for x in models):
            ct_invalid += 1
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
            all_ips[ip] = {"ip": ip, "count": 0, "sanitized_id": len(all_ips)}
        all_ips[ip]["count"] += 1
        if sanitize_ip:
            user_id = f"arena_user_{all_ips[ip]['sanitized_id']}"
        else:
            user_id = f"{all_ips[ip]['ip']}"

        if ban_ip_list is not None and ip in ban_ip_list:
            ct_banned += 1
            continue

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
                anony=anony,
                language=lang_code,
                tstamp=row["tstamp"],
            )
        )

        all_models.update(models_hidden)
    battles.sort(key=lambda x: x["tstamp"])
    last_updated_tstamp = battles[-1]["tstamp"]

    last_updated_datetime = datetime.datetime.fromtimestamp(
        last_updated_tstamp, tz=timezone("US/Pacific")
    ).strftime("%Y-%m-%d %H:%M:%S %Z")

    print(
        f"#votes: {len(data)}, #invalid votes: {ct_invalid}, "
        f"#leaked_identity: {ct_leaked_identity} "
        f"#banned: {ct_banned} "
    )
    print(f"#battles: {len(battles)}, #anony: {ct_anony}")
    print(f"#models: {len(all_models)}, {all_models}")
    print(f"last-updated: {last_updated_datetime}")

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

    with open(output, "w") as fout:
        json.dump(battles, fout, indent=2, ensure_ascii=False)
    print(f"Write cleaned data to {output}")
