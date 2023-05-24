import argparse
from collections import defaultdict
import datetime
import json
import math
import pickle
from pytz import timezone

import gdown
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm

from fastchat.model.model_registry import get_model_info
from fastchat.serve.monitor.basic_stats import get_log_files
from fastchat.serve.monitor.clean_battle_data import clean_battle_data


pd.options.display.float_format = "{:.2f}".format


def compute_elo(battles, K=4, SCALE=400, BASE=10, INIT_RATING=1000):
    rating = defaultdict(lambda: INIT_RATING)

    for rd, model_a, model_b, win in battles[
        ["model_a", "model_b", "win"]
    ].itertuples():
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


def get_bootstrap_result(battles, func_compute_elo, num_round=1000):
    rows = []
    for i in tqdm(range(num_round), desc="bootstrap"):
        tmp_battles = battles.sample(frac=1.0, replace=True)
        # tmp_battles = tmp_battles.sort_values(ascending=True, by=["tstamp"])
        rows.append(func_compute_elo(tmp_battles))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


def get_elo_from_bootstrap(bootstrap_df):
    return dict(bootstrap_df.quantile(0.5))


def compute_pairwise_win_fraction(battles, model_order):
    # Times each model wins as Model A
    a_win_ptbl = pd.pivot_table(
        battles[battles["win"] == "model_a"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )

    # Table counting times each model wins as Model B
    b_win_ptbl = pd.pivot_table(
        battles[battles["win"] == "model_b"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )

    # Table counting number of A-B pairs
    num_battles_ptbl = pd.pivot_table(
        battles, index="model_a", columns="model_b", aggfunc="size", fill_value=0
    )

    # Computing the proportion of wins for each model as A and as B
    # against all other models
    row_beats_col_freq = (a_win_ptbl + b_win_ptbl.T) / (
        num_battles_ptbl + num_battles_ptbl.T
    )

    if model_order is None:
        prop_wins = row_beats_col_freq.mean(axis=1).sort_values(ascending=False)
        model_order = list(prop_wins.keys())

    # Arrange ordering according to proprition of wins
    row_beats_col = row_beats_col_freq.loc[model_order, model_order]
    return row_beats_col


def visualize_leaderboard_table(rating):
    models = list(rating.keys())
    models.sort(key=lambda k: -rating[k])

    emoji_dict = {
        1: "🥇",
        2: "🥈",
        3: "🥉",
    }

    md = ""
    md += "| Rank | Model | Elo Rating | Description |\n"
    md += "| --- | --- | --- | --- |\n"
    for i, model in enumerate(models):
        rank = i + 1
        minfo = get_model_info(model)
        emoji = emoji_dict.get(rank, "")
        md += f"| {rank} | {emoji} [{model}]({minfo.link}) | {rating[model]:.0f} | {minfo.description} |\n"

    return md


def visualize_pairwise_win_fraction(battles, model_order):
    row_beats_col = compute_pairwise_win_fraction(battles, model_order)
    fig = px.imshow(
        row_beats_col,
        color_continuous_scale="RdBu",
        text_auto=".2f",
        height=600,
        width=600,
    )
    fig.update_layout(
        xaxis_title="Model B",
        yaxis_title="Model A",
        xaxis_side="top",
        title_y=0.07,
        title_x=0.5,
    )
    fig.update_traces(
        hovertemplate="Model A: %{y}<br>Model B: %{x}<br>Fraction of A Wins: %{z}<extra></extra>"
    )

    return fig


def visualize_battle_count(battles, model_order):
    ptbl = pd.pivot_table(
        battles, index="model_a", columns="model_b", aggfunc="size", fill_value=0
    )
    battle_counts = ptbl + ptbl.T
    fig = px.imshow(
        battle_counts.loc[model_order, model_order],
        text_auto=True,
        height=600,
        width=600,
    )
    fig.update_layout(
        xaxis_title="Model B",
        yaxis_title="Model A",
        xaxis_side="top",
        title_y=0.07,
        title_x=0.5,
    )
    fig.update_traces(
        hovertemplate="Model A: %{y}<br>Model B: %{x}<br>Count: %{z}<extra></extra>"
    )
    return fig


def visualize_average_win_rate(battles):
    row_beats_col_freq = compute_pairwise_win_fraction(battles, None)
    fig = px.bar(
        row_beats_col_freq.mean(axis=1).sort_values(ascending=False),
        text_auto=".2f",
        height=400,
        width=600,
    )
    fig.update_layout(
        yaxis_title="Average Win Rate", xaxis_title="Model", showlegend=False
    )
    return fig


def visualize_bootstrap_elo_rating(df):
    bars = (
        pd.DataFrame(
            dict(
                lower=df.quantile(0.025),
                rating=df.quantile(0.5),
                upper=df.quantile(0.975),
            )
        )
        .reset_index(names="model")
        .sort_values("rating", ascending=False)
    )
    bars["error_y"] = bars["upper"] - bars["rating"]
    bars["error_y_minus"] = bars["rating"] - bars["lower"]
    bars["rating_rounded"] = np.round(bars["rating"], 2)
    fig = px.scatter(
        bars,
        x="model",
        y="rating",
        error_y="error_y",
        error_y_minus="error_y_minus",
        text="rating_rounded",
        height=400,
        width=600,
    )
    fig.update_layout(xaxis_title="Model", yaxis_title="Rating")
    return fig


def report_elo_analysis_results(battles_json):
    battles = pd.DataFrame(battles_json)
    battles = battles.sort_values(ascending=True, by=["tstamp"])
    # Only use anonymous votes
    battles = battles[battles["anony"]].reset_index(drop=True)
    battles_no_ties = battles[~battles["win"].str.contains("tie")]

    # Online update
    elo_rating_online = compute_elo(battles)

    # Bootstrap
    bootstrap_df = get_bootstrap_result(battles, compute_elo)
    elo_rating_median = get_elo_from_bootstrap(bootstrap_df)
    elo_rating_median = {k: int(v + 0.5) for k, v in elo_rating_median.items()}
    model_order = list(elo_rating_online.keys())
    model_order.sort(key=lambda k: -elo_rating_online[k])

    # Plots
    leaderboard_table = visualize_leaderboard_table(elo_rating_online)
    win_fraction_heatmap = visualize_pairwise_win_fraction(battles_no_ties, model_order)
    battle_count_heatmap = visualize_battle_count(battles_no_ties, model_order)
    average_win_rate_bar = visualize_average_win_rate(battles_no_ties)
    bootstrap_elo_rating = visualize_bootstrap_elo_rating(bootstrap_df)

    last_updated_tstamp = battles["tstamp"].max()
    last_updated_datetime = datetime.datetime.fromtimestamp(
        last_updated_tstamp, tz=timezone("US/Pacific")
    ).strftime("%Y-%m-%d %H:%M:%S %Z")

    return {
        "elo_rating_online": elo_rating_online,
        "elo_rating_median": elo_rating_median,
        "leaderboard_table": leaderboard_table,
        "win_fraction_heatmap": win_fraction_heatmap,
        "battle_count_heatmap": battle_count_heatmap,
        "average_win_rate_bar": average_win_rate_bar,
        "bootstrap_elo_rating": bootstrap_elo_rating,
        "last_updated_datetime": last_updated_datetime,
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

    print("# Online")
    pretty_print_elo_rating(results["elo_rating_online"])
    print("# Median")
    pretty_print_elo_rating(results["elo_rating_median"])
    print(f"last update : {results['last_updated_datetime']}")

    with open("elo_results.pkl", "wb") as fout:
        pickle.dump(results, fout)
