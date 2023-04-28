import argparse
import json

import polyglot
from polyglot.detect import Detector
import pycld2
from tqdm import tqdm

from fastchat.serve.monitor.basic_stats import get_log_files


VOTES = ["tievote", "leftvote", "rightvote", "bothbad_vote"]

def detect_lang(text):
    try:
        lang_code = Detector(text).language.name
    except (pycld2.error, polyglot.detect.base.UnknownLanguage):
        lang_code = "unknown"
    return lang_code


def remove_html(raw):
    if raw.startswith("<h3>"):
        return raw[raw.find(": ") + 2: -len('</h3>\n')]
    return raw


def clean_battle_data(log_files):
    data = []
    for filename in tqdm(log_files):
        with open(filename) as f:
            lines = f.readlines()
        for l in lines:
            dp = json.loads(l)
            if dp["type"] in VOTES:
                data.append(dp)

    convert_type = {
        "leftvote": "model_a",
        "rightvote": "model_b",
        "tievote": "tie",
        "bothbad_vote": "tie (bothbad)",
    }

    all_models = set()
    ct_annoy = 0
    ct_invalid = 0
    battles = []
    for row in data:
        models_public = [remove_html(row["models"][0]), remove_html(row["models"][1])]
        if "model_name" in row["states"][0]:
            models_hidden = [row["states"][0]["model_name"], row["states"][1]["model_name"]]
            if models_hidden[0] is None:
                models_hidden = models_public
        else:
            models_hidden = models_public

        if ((models_public[0] == "" and models_public[1] != "") or
            (models_public[1] == "" and models_public[0] != "")):
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

        state = row["states"][0]
        lang_code = detect_lang(state["messages"][state["offset"]][1])

        battles.append(dict(
            model_a=models[0],
            model_b=models[1],
            win=convert_type[row["type"]],
            anony=anony,
            tstamp=row["tstamp"],
            language=lang_code,
        ))

        all_models.update(models_hidden)

    print(f"#votes: {len(data)}, #invalid votes: {ct_invalid}")
    print(f"#battles: {len(battles)}, #annoy: {ct_annoy}")
    print(f"#models: {len(all_models)}, {all_models}")

    return battles


if __name__ == "__main__":
    log_files = get_log_files()
    battles = clean_battle_data(log_files)

    print("Samples:")
    for i in range(4):
        print(battles[i])

    json.dump(battles, open("clean_battle.json", "w"), indent=2)
