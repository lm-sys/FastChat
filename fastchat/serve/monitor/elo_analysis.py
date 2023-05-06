import argparse
from collections import defaultdict
import json
import math
import pickle

import gdown
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm

from fastchat.serve.monitor.basic_stats import get_log_files
from fastchat.serve.monitor.clean_battle_data import clean_battle_data
from fastchat.serve.gradio_web_server import model_info


pd.options.display.float_format = "{:.2f}".format


def compute_elo(battles, K=32, SCALE=400, BASE=10, INIT_RATING=1000):
    rating = defaultdict(lambda: INIT_RATING)
 
    for rd, model_a, model_b, win in battles[['model_a', 'model_b', 'win']].itertuples():
        ra = rating[model_a]
        rb = rating[model_b]
        ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
        eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))
        if win == "model_a":
            sa = 1
        elif win == "model_b":
            sa = 0
        elif win == "tie" or win == "tie (bothbad)":
            sa = 0.5
        else:
            raise Exception(f"unexpected vote {win}")
        rating[model_a] += K * (sa - ea)
        rating[model_b] += K * (1 - sa - eb)
    
    return dict(rating)


def get_bootstrap_result(battles, func_compute_elo, num_round):
    rows = []
    for i in tqdm(range(num_round), desc="bootstrap"):
        rows.append(func_compute_elo(battles.sample(frac=1.0, replace=True)))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


def visualize_leaderboard_md(rating):
    models = list(rating.keys())
    models.sort(key=lambda k: -rating[k])

    emoji_dict = {
        1: "🥇",
        2: "🥈",
        3: "🥉",
    }
   
    md = """
# Leaderboard
[[Blog](https://lmsys.org/blog/2023-05-03-arena/)] [[GitHub]](https://github.com/lm-sys/FastChat) [[Twitter]](https://twitter.com/lmsysorg) [[Discord]](https://discord.gg/h6kCZb72G7)

We use the Elo rating system to calculate the relative performance of the models. You can view the voting data, basic analyses, and calculation procedure in this [notebook](https://colab.research.google.com/drive/1lAQ9cKVErXI1rEYq7hTKNaCQ5Q8TzrI5?usp=sharing). The current leaderboard is based on the data we collected before May 1, 2023. We will periodically release new leaderboards.\n
"""
    md += "| Rank | Model | Elo Rating | Description |\n"
    md += "| --- | --- | --- | --- |\n"
    for i, model in enumerate(models):
        rank = i + 1
        _, link, desc = model_info[model]
        emoji = emoji_dict.get(rank, "")
        md += f"| {rank} | {emoji} [{model}]({link}) | {rating[model]:.0f} | {desc} |\n"

    return md

def visualize_bootstrap_elo_rating(battles, num_round=1000):
    df = get_bootstrap_result(battles, compute_elo, num_round)

    bars = pd.DataFrame(dict(
        lower = df.quantile(.025),
        rating = df.quantile(.5),
        upper = df.quantile(.975))).reset_index(names="model").sort_values("rating", ascending=False)
    bars['error_y'] = bars['upper'] - bars["rating"]
    bars['error_y_minus'] = bars['rating'] - bars["lower"]
    bars['rating_rounded'] = np.round(bars['rating'], 2)
    fig = px.scatter(bars, x="model", y="rating", error_y="error_y", 
                     error_y_minus="error_y_minus", text="rating_rounded",
                     width=450)
    fig.update_layout(xaxis_title="Model", yaxis_title="Rating")
    return fig


def compute_pairwise_win_fraction(battles, model_order):
    # Times each model wins as Model A
    a_win_ptbl = pd.pivot_table(
        battles[battles['win'] == "model_a"], 
        index="model_a", columns="model_b", aggfunc="size", fill_value=0)

    # Table counting times each model wins as Model B
    b_win_ptbl = pd.pivot_table(
        battles[battles['win'] == "model_b"], 
        index="model_a", columns="model_b", aggfunc="size", fill_value=0)

    # Table counting number of A-B pairs
    num_battles_ptbl = pd.pivot_table(battles, 
        index="model_a", columns="model_b", aggfunc="size", fill_value=0)

    # Computing the proportion of wins for each model as A and as B 
    # against all other models
    row_beats_col_freq = (
        (a_win_ptbl + b_win_ptbl.T) / 
        (num_battles_ptbl + num_battles_ptbl.T)
    )

    if model_order is None:
        prop_wins = row_beats_col_freq.mean(axis=1).sort_values(ascending=False)
        model_order = list(prop_wins.keys())

    # Arrange ordering according to proprition of wins
    row_beats_col = row_beats_col_freq.loc[model_order, model_order]
    return row_beats_col
  

def visualize_pairwise_win_fraction(battles, model_order):
    row_beats_col = compute_pairwise_win_fraction(battles, model_order)
    fig = px.imshow(row_beats_col, color_continuous_scale='RdBu',
                    text_auto=".2f", height=500, width=500)
    fig.update_layout(xaxis_title="Model B", 
                  yaxis_title="Model A",
                  xaxis_side="top",
                  title_y=0.07, title_x=0.5)
    fig.update_traces(hovertemplate=
        "Model A: %{y}<br>Model B: %{x}<br>Fraction of A Wins: %{z}<extra></extra>")

    return fig


def visualize_battle_count(battles, model_order):
    ptbl = pd.pivot_table(battles, index="model_a", columns="model_b", aggfunc="size", 
                          fill_value=0)
    battle_counts = ptbl + ptbl.T
    fig = px.imshow(battle_counts.loc[model_order, model_order], text_auto=True,
                    height=500, width=500)
    fig.update_layout(xaxis_title="Model B", 
                      yaxis_title="Model A",
                      xaxis_side="top",
                      title_y=0.07, title_x=0.5)
    fig.update_traces(hovertemplate=
                      "Model A: %{y}<br>Model B: %{x}<br>Count: %{z}<extra></extra>")
    return fig


def visualize_average_win_rate(battles):
    row_beats_col_freq = compute_pairwise_win_fraction(battles, None)
    fig = px.bar(row_beats_col_freq.mean(axis=1).sort_values(ascending=False),
                 text_auto=".2f", width=450)
    fig.update_layout(yaxis_title="Average Win Rate", xaxis_title="Model",
                      showlegend=False)
    return fig


def report_elo_analysis_results(battles_json):
    battles = pd.DataFrame(battles_json)
    battles = battles.sort_values(ascending=True, by=["tstamp"])
    # Only use anonymous votes
    battles = battles[battles["anony"]].reset_index(drop=True)
    battles_no_ties = battles[~battles["win"].str.contains("tie")]

    elo_rating = compute_elo(battles)
    elo_rating = {k: int(v) for k, v in elo_rating.items()}

    model_order = list(elo_rating.keys())
    model_order.sort(key=lambda k: -elo_rating[k])

    leaderboard_md = visualize_leaderboard_md(elo_rating)
    win_fraction_heatmap = visualize_pairwise_win_fraction(battles_no_ties, model_order)
    battle_count_heatmap = visualize_battle_count(battles_no_ties, model_order)
    average_win_rate_bar = visualize_average_win_rate(battles_no_ties)
    bootstrap_elo_rating = visualize_bootstrap_elo_rating(battles)

    return {
        "elo_rating": elo_rating,
        "leaderboard_md": leaderboard_md,
        "win_fraction_heatmap": win_fraction_heatmap,
        "battle_count_heatmap": battle_count_heatmap,
        "average_win_rate_bar": average_win_rate_bar,
        "bootstrap_elo_rating": bootstrap_elo_rating,
    }


def pretty_print_elo_rating(rating):
    model_order = list(rating.keys())
    model_order.sort(key=lambda k: -rating[k])
    for i, model in enumerate(model_order):
        print(f"{i+1:2d}, {model:25s}, {rating[model]:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-battle-file", type=str)
    parser.add_argument("--max-num-files", type=int)
    args = parser.parse_args()

    if args.clean_battle_file:
        # Read data from a cleaned battle files
        battles = pd.read_json(args.clean_battle_file)
    else:
        # Read data from all log files
        log_files = get_log_files(args.max_num_files)
        battles = clean_battle_data(log_files)

    results = report_elo_analysis_results(battles)

    pretty_print_elo_rating(results["elo_rating"])

    with open("elo_results.pkl", "wb") as fout:
        pickle.dump(results, fout)
