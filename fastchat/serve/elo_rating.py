import json
import numpy as np
import os
import pandas as pd
import plotly.express as px
from tqdm import tqdm
from typing import List, Dict

MODELS = ["vicuna-13b", "alpaca-13b", "dolly-v2-12b", "oasst-pythia-12b",
          "koala-13b", "llama-13b", "stablelm-tuned-alpha-7b", "chatglm-6b"]
VOTES = ["tievote", "leftvote", "rightvote", "share"]
log_dir = "/Users/Ying/work/project/chatbot/arena_logs"
elo_k = 5
BOOTSTRAP_ROUNDS = 100


def unique(models: List):
    return tuple(sorted(models))


def remove_html(raw):
    if raw.startswith("<h3>"):
        return raw[raw.find(": ") + 2: -len('</h3>\n')]
    return raw


def compute_elo(battles):
    rating = {}
    for model in MODELS:
        rating[model] = 1000
 
    for rd, (models, vote) in enumerate(battles):
        p1 = rating[models[0]] / (rating[models[0]] + rating[models[1]])
        p2 = rating[models[1]] / (rating[models[0]] + rating[models[1]])
        if vote == "leftvote":
            rating[models[0]] += elo_k * (1 - p1)
            rating[models[1]] -= elo_k * p2
        if vote == "rightvote":
            rating[models[0]] -= elo_k * p1
            rating[models[1]] += elo_k * (1 - p2)
    
        # if rd % 100 == 0:
        #     print("=" * 30, rd, ["battles", "anonymous", "normalized"][i], "=" * 30)
        #     for model in MODELS:
        #         print(f"{model}: {rating[i][model]:.2f}")
    return rating


def get_bootstrap_result(battles: List[Dict]):
    df = pd.DataFrame(battles)
    rows = []
    for i in tqdm(range(BOOTSTRAP_ROUNDS), desc="bootstrap"):
        subset = df.sample(frac=1, replace=True).values
        elo = compute_elo(subset)
        rows.append(elo)
    return pd.DataFrame(rows)


def plot_bootstrap_scores(df, title, outfile=None):
    bars = pd.DataFrame(dict(
        lower = df.quantile(.025),
        score = df.quantile(.5),
        upper = df.quantile(.975))).reset_index().sort_values("score", ascending=False)
    bars['error_y'] = bars['upper'] - bars["score"]
    bars['error_y_minus'] = bars['score'] - bars["lower"]
    bars = bars.rename(columns={"index": "model"})
    if outfile is None:
        print("=" * 20, "bootstrap scores", "=" * 20)
        print(bars[["model", "score", "lower", "upper", "error_y_minus", "error_y"]])
    else:
        outfile.write("## bootstrap scores\n")
        outfile.write(bars[["model", "score", "lower", "upper", "error_y_minus", "error_y"]].to_markdown())
    return px.scatter(bars, x="model", y="score", error_y="error_y",
                      error_y_minus="error_y_minus",
                      title=title, height = 600)
    

if __name__ == "__main__":
    data = []
    for i in range(10):
        for day in ["24", "25"]:
            filename = os.path.join(log_dir, f"server{i}_2023-04-{day}-conv.json")
            with open(filename) as f:
                lines = f.readlines()
            for l in lines:
                dp = json.loads(l)
                if "models" in dp and dp["type"] != "bothbad_vote":
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
            if unique(models) not in battle_cnt:
                battle_cnt[unique(models)] = 0
            battle_cnt[unique(models)] += 1
        battles.append((models, x["type"]))
    print("number of model name inconsistent rounds", inconsist)
    print("battle pair counts", sorted(battle_cnt.values()))
    print("number of occurred pairs", len(battle_cnt))

    ub = min(battle_cnt.values()) * 1.5
    normalized = []
    cur_cnt = {}
    for models, vote in anonymous:
        if unique(models) not in cur_cnt:
            cur_cnt[unique(models)] = 0
        if cur_cnt[unique(models)] < ub:
            normalized.append((models, vote))
            cur_cnt[unique(models)] += 1

    print("len(battles), len(anonymous), len(normalized):",
          len(battles), len(anonymous), len(normalized))

    # get ratings
    rating = [None] * 3
    for i, subset in enumerate([battles, anonymous, normalized]):
        rating[i] = compute_elo(subset)

        print("=" * 30, len(subset), ["battles", "anonymous", "normalized"][i], "=" * 30)
        leaderboard = sorted(rating[i].items(), key=lambda item: -item [1])
        for x in leaderboard:
            print(f"{x[0]}: {x[1]:.2f}")

    # get bootstrap result
    ratings = get_bootstrap_result(anonymous)
    fig = plot_bootstrap_scores(ratings, "boostrap elo")
    fig.show()

 
