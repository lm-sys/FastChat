import argparse
from collections import defaultdict
import json
import math
import os
from typing import List, Dict

import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm

from fastchat.serve.monitor.basic_stats import get_log_files


MODELS = ["vicuna-13b", "alpaca-13b", "dolly-v2-12b", "oasst-pythia-12b",
          "koala-13b", "llama-13b", "stablelm-tuned-alpha-7b", "chatglm-6b"]
VOTES = ["tievote", "leftvote", "rightvote"]
BOOTSTRAP_ROUNDS = 100

INIT_RATING = 1000
BASE = 10
SCALE = 400
K = 32


# TODO compute likelihood


def unique(models: List):
    return tuple(sorted(models))


def remove_html(raw):
    if raw.startswith("<h3>"):
        return raw[raw.find(": ") + 2: -len('</h3>\n')]
    return raw


def collect_data(log_files):
    data = []
    for filename in log_files:
        with open(filename) as f:
            lines = f.readlines()
        for l in lines:
            dp = json.loads(l)
            if "models" in dp and dp["type"] in VOTES:
                data.append(dp)
    print("number of rounds", len(data))

    # collect battle data
    battles = []
    anonymous = []
    battle_cnt = {}
    inconsist = 0
    for x in data:
        models = [remove_html(x["models"][0]), remove_html(x["models"][1])]
        if "model_name" in x["states"][0]:
            models2 = [x["states"][0]["model_name"], x["states"][1]["model_name"]]
            if models[0] in MODELS and models2[0] in MODELS:
                if not models == models2:
                    inconsist += 1
                    # print(x)
                    # print("=" * 60)
                    continue
            if models2[0] not in MODELS:
                assert models2[0] is None
            if models[0] not in MODELS:
                assert models[0] == '' or models[0] == "Model A"
        assert models[0] in MODELS or models2[0] in MODELS
        assert x["type"] in VOTES

        if not models[0] in MODELS:
            models = models2
            anonymous.append((models, x["type"]))
            um = unique(models)
            if um not in battle_cnt:
                battle_cnt[um] = 0
            battle_cnt[um] += 1

        battles.append((models, x["type"]))
    print("#inconsistent model name rounds", inconsist)
    print("anony. battle pair counts", sorted(battle_cnt.values()))
    print("#battles", len(battle_cnt))

    ub = min(battle_cnt.values()) * 1.5
    normalized = []
    cur_cnt = {}
    perm = np.random.permutation(len(anonymous))
    anonymous = [anonymous[i] for i in perm]
    for models, vote in anonymous:
        um = unique(models)
        if um not in cur_cnt:
            cur_cnt[um] = 0
        if cur_cnt[um] < ub:
            normalized.append((models, vote))
            cur_cnt[um] += 1

    print("len(battles), len(anonymous), len(normalized):",
          len(battles), len(anonymous), len(normalized))
    return battles, anonymous, normalized


def compute_elo(battles):
    rating = {}
    for model in MODELS:
        rating[model] = INIT_RATING
 
    for rd, (models, vote) in enumerate(battles):
        ra = rating[models[0]]
        rb = rating[models[1]]
        ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
        eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))
        # linear update
        if vote == "leftvote":
            sa = 1
        elif vote == "rightvote":
            sa = 0
        elif vote == "tievote":
            sa = 0.5
        else:
            raise Exception(f"unexpected vote {vote}")
        rating[models[0]] += K * (sa - ea)
        rating[models[1]] += K * (1 - sa - eb)
   
        # if rd % 100 == 0:
        #     print("=" * 30, rd, ["battles", "anonymous", "normalized"][i], "=" * 30)
        #     for model in MODELS:
        #         print(f"{model}: {rating[i][model]:.2f}")
    return rating


def get_bootstrap_result(battles: List[Dict]):
    df = pd.DataFrame(battles)
    rows = []
    for i in tqdm(range(BOOTSTRAP_ROUNDS), desc="bootstrap"):
        subset = df.sample(frac=0.95, replace=True).values
        elo = compute_elo(subset)
        rows.append(elo)
    return pd.DataFrame(rows)


def get_likelihood(ratings, battles):
    llh = 0
    for rd, (models, vote) in enumerate(battles):
        ra = ratings.loc[ratings["model"] == models[0], "score"].item()
        rb = ratings.loc[ratings["model"] == models[1], "score"].item()
        ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
        eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))
        if vote == "leftvote":
            llh += math.log(ea) / len(battles)
        elif vote == "rightvote":
            llh += math.log(eb) / len(battles)
        elif vote == "tievote":
            llh += math.log(0.5) / len(battles)
    return llh


def plot_bootstrap_scores(df, battles, title):
    bars = pd.DataFrame(dict(
        lower = df.quantile(.025),
        score = df.quantile(.5),
        upper = df.quantile(.975))).reset_index().sort_values("score", ascending=False)
    bars['error_y'] = bars['upper'] - bars["score"]
    bars['error_y_minus'] = bars['score'] - bars["lower"]
    bars = bars.rename(columns={"index": "model"})
    bars['rank'] = range(1, len(bars) + 1)

    llh = get_likelihood(bars[["model", "score"]], battles)

    indices = ["score", "lower", "upper"]
    bars[indices] = bars[indices].astype(int)

    ret_md = bars[["rank", "model", "score", "lower", "upper"]
        ].to_markdown(index=False, tablefmt="github")
    #px.scatter(bars, x="model", y="score", error_y="error_y",
    #          error_y_minus="error_y_minus", title=title, height = 600)
    return {
        "log_likelihood": llh,
        "md": ret_md,
    }


def compute_coverage(rows):
    pairs = defaultdict(lambda: defaultdict(lambda :0))
    names = set()
    for models, _ in rows:
        models = unique(models)
        pairs[models[0]][models[1]] += 1
        names.add(models[0])
        names.add(models[1])

    names = sorted(names)
    data = {
        left: [pairs[left].get(right, "") for right in names]
        for left in names
    }

    df = pd.DataFrame(data, index=names).reset_index()
    df = df.rename(columns={"index": "model"})
    return df


def report_ratings_linear_update(log_files):
    battles, anonymous, normalized = collect_data(log_files)

    # Plot coverage
    battles_cov = compute_coverage(battles)
    battles_cov_md = battles_cov.to_markdown(index=False, tablefmt="github")

    anony_cov = compute_coverage(anonymous)
    anony_cov_md = anony_cov.to_markdown(index=False, tablefmt="github")

    # Get ratings
    rating = [None] * 3
    for i, subset in enumerate([battles, anonymous, normalized]):
        rating[i] = compute_elo(subset)
        print("=" * 30, len(subset), ["battles", "anonymous", "normalized"][i], "=" * 30)
        leaderboard = sorted(rating[i].items(), key=lambda item: -item [1])
        for x in leaderboard:
            print(f"{x[0]}: {x[1]:.2f}")

    # Get bootstrap result
    ratings = get_bootstrap_result(anonymous)
    plots = plot_bootstrap_scores(ratings, anonymous, "boostrap elo")
    return {
        #"battles_cov_md": battles_cov_md,
        "anony_cov_md": anony_cov_md,
        "anony_rating_md": plots["md"],
        "anony_log_likelihood": plots["log_likelihood"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    np.random.seed(0)

    log_files = get_log_files()
    elo_ratings = report_ratings_linear_update(log_files)
    print(elo_ratings["anony_cov_md"])

    print(f"log-likelihood: {elo_ratings['anony_log_likelihood']:.2f}")
    print(elo_ratings["anony_rating_md"])
