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
    "openai",
    "anthropic",
    "claude",
    "bard",
    "palm",
    "lamda",
    "google",
    "llama",
    "NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.",
    "$MODERATION$ YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES.",
]

for i in range(len(IDENTITY_WORDS)):
    IDENTITY_WORDS[i] = IDENTITY_WORDS[i].lower()


def get_log_files(max_num_files=None):
    dates = []
    for month in range(4, 12):
        for day in range(1, 33):
            dates.append(f"2023-{month:02d}-{day:02d}")

    filenames = []
    for d in dates:
        for i in range(NUM_SERVERS):
            name = os.path.expanduser(f"~/fastchat_logs/server{i}/{d}-conv.json")
            if os.path.exists(name):
                filenames.append(name)
    max_num_files = max_num_files or len(filenames)
    filenames = filenames[-max_num_files:]
    return filenames


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


def replace_model_name(old_name):
    return (
        old_name.replace("bard", "palm-2")
        .replace("claude-v1", "claude-1")
        .replace("claude-instant-v1", "claude-instant-1")
        .replace("oasst-sft-1-pythia-12b", "oasst-pythia-12b")
    )


def clean_battle_data(log_files, exclude_model_names):
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
            if row["type"] in VOTES:
                data.append(row)

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
            for role, msg in state["messages"][state["offset"] :]:
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
        models = [replace_model_name(m) for m in models]

        # Exclude certain models
        if any(x in exclude_model_names for x in models):
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
            all_ips[ip] = len(all_ips)
        user_id = all_ips[ip]

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
        f"#leaked_identity: {ct_leaked_identity}"
    )
    print(f"#battles: {len(battles)}, #anony: {ct_anony}")
    print(f"#models: {len(all_models)}, {all_models}")
    print(f"last-updated: {last_updated_datetime}")

    return battles


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-num-files", type=int)
    parser.add_argument(
        "--mode", type=str, choices=["simple", "conv_release"], default="simple"
    )
    parser.add_argument("--exclude-model-names", type=str, nargs="+")
    args = parser.parse_args()

    log_files = get_log_files(args.max_num_files)
    battles = clean_battle_data(log_files, args.exclude_model_names or [])
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
