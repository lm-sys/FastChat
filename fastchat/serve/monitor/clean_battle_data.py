import argparse
import datetime
import json
from pytz import timezone
import os
import time

from tqdm import tqdm

from fastchat.serve.monitor.basic_stats import get_log_files


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
]


def get_log_files(max_num_files=None):
    dates = []
    for month in [4]:
        for day in range(24, 32):
            dates.append(f"2023-{month:02d}-{day:02d}")
    for month in [5]:
        for day in range(1, 9):
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


def detect_lang(text):
    import polyglot
    from polyglot.detect import Detector
    from polyglot.detect.base import logger as polyglot_logger
    import pycld2

    polyglot_logger.setLevel("ERROR")

    try:
        lang_code = Detector(text).language.name
    except (pycld2.error, polyglot.detect.base.UnknownLanguage):
        lang_code = "unknown"
    return lang_code


def remove_html(raw):
    if raw.startswith("<h3>"):
        return raw[raw.find(": ") + 2 : -len("</h3>\n")]
    return raw


def clean_battle_data(log_files):
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
    ct_annoy = 0
    ct_invalid = 0
    ct_leaked_identity = 0
    battles = []
    for row in data:
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
            ct_annoy += 1
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
        lang_code = detect_lang(state["messages"][state["offset"]][1])
        rounds = (len(state["messages"]) - state["offset"]) // 2

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

        # Keep the result
        battles.append(
            dict(
                model_a=models[0],
                model_b=models[1],
                win=convert_type[row["type"]],
                anony=anony,
                rounds=rounds,
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
    print(f"#battles: {len(battles)}, #annoy: {ct_annoy}")
    print(f"#models: {len(all_models)}, {all_models}")
    print(f"last-updated: {last_updated_datetime}")

    return battles


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-num-files", type=int)
    args = parser.parse_args()

    log_files = get_log_files(args.max_num_files)
    battles = clean_battle_data(log_files)

    print("Samples:")
    for i in range(4):
        print(battles[i])

    date = datetime.datetime.now(tz=timezone("US/Pacific")).strftime("%Y%m%d")
    output = f"clean_battle_{date}.json"
    with open(output, "w") as fout:
        json.dump(battles, fout, indent=2)
    print(f"Write cleaned data to {output}")
