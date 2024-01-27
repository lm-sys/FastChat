import argparse
from collections import defaultdict
import datetime
import json
import math
import pickle
from pytz import timezone

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

    for rd, model_a, model_b, winner in battles[
        ["model_a", "model_b", "winner"]
    ].itertuples():
        ra = rating[model_a]
        rb = rating[model_b]
        ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
        eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))
        if winner == "model_a":
            sa = 1
        elif winner == "model_b":
            sa = 0
        elif winner == "tie" or winner == "tie (bothbad)":
            sa = 0.5
        else:
            raise Exception(f"unexpected vote {winner}")
        rating[model_a] += K * (sa - ea)
        rating[model_b] += K * (1 - sa - eb)

    return dict(rating)


def get_bootstrap_result(battles, func_compute_elo, num_round=1000):
    rows = []
    for i in tqdm(range(num_round), desc="bootstrap"):
        tmp_battles = battles.sample(frac=1.0, replace=True)
        rows.append(func_compute_elo(tmp_battles))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


def compute_elo_mle_with_tie(df, SCALE=400, BASE=10, INIT_RATING=1000):
    from sklearn.linear_model import LogisticRegression

    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # duplicate battles
    df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df["model_a"]]] = +math.log(BASE)
    X[np.arange(n), models[df["model_b"]]] = -math.log(BASE)

    # one A win => two A win
    Y = np.zeros(n)
    Y[df["winner"] == "model_a"] = 1.0

    # one tie => one A win + one B win
    # find tie + tie (both bad) index
    tie_idx = (df["winner"] == "tie") | (df["winner"] == "tie (bothbad)")
    tie_idx[len(tie_idx) // 2 :] = False
    Y[tie_idx] = 1.0

    lr = LogisticRegression(fit_intercept=False)
    lr.fit(X, Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING
    # calibrate llama-13b to 800 if applicable
    if "llama-13b" in models.index:
        elo_scores += 800 - elo_scores[models["llama-13b"]]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def get_median_elo_from_bootstrap(bootstrap_df):
    median = dict(bootstrap_df.quantile(0.5))
    median = {k: int(v + 0.5) for k, v in median.items()}
    return median


def compute_pairwise_win_fraction(battles, model_order, limit_show_number=None):
    # Times each model wins as Model A
    a_win_ptbl = pd.pivot_table(
        battles[battles["winner"] == "model_a"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )

    # Table counting times each model wins as Model B
    b_win_ptbl = pd.pivot_table(
        battles[battles["winner"] == "model_b"],
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

    if limit_show_number is not None:
        model_order = model_order[:limit_show_number]

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
        height=700,
        width=700,
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
        height=700,
        width=700,
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


def visualize_average_win_rate(battles, limit_show_number):
    row_beats_col_freq = compute_pairwise_win_fraction(
        battles, None, limit_show_number=limit_show_number
    )
    fig = px.bar(
        row_beats_col_freq.mean(axis=1).sort_values(ascending=False),
        text_auto=".2f",
        height=500,
        width=700,
    )
    fig.update_layout(
        yaxis_title="Average Win Rate", xaxis_title="Model", showlegend=False
    )
    return fig


def visualize_bootstrap_elo_rating(df, df_final, limit_show_number):
    bars = (
        pd.DataFrame(
            dict(
                lower=df.quantile(0.025),
                rating=df_final,
                upper=df.quantile(0.975),
            )
        )
        .reset_index(names="model")
        .sort_values("rating", ascending=False)
    )
    bars = bars[:limit_show_number]
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
        height=500,
        width=700,
    )
    fig.update_layout(xaxis_title="Model", yaxis_title="Rating")
    return fig


def report_elo_analysis_results(battles_json, rating_system="bt", num_bootstrap=100):
    battles = pd.DataFrame(battles_json)
    battles = battles.sort_values(ascending=True, by=["tstamp"])
    # Only use anonymous votes
    battles = battles[battles["anony"]].reset_index(drop=True)
    battles_no_ties = battles[~battles["winner"].str.contains("tie")]

    # Online update
    elo_rating_online = compute_elo(battles)

    if rating_system == "bt":
        bootstrap_df = get_bootstrap_result(
            battles, compute_elo_mle_with_tie, num_round=num_bootstrap
        )
        elo_rating_final = compute_elo_mle_with_tie(battles)
    elif rating_system == "elo":
        bootstrap_df = get_bootstrap_result(
            battles, compute_elo, num_round=num_bootstrap
        )
        elo_rating_median = get_median_elo_from_bootstrap(bootstrap_df)
        elo_rating_final = elo_rating_median

    model_order = list(elo_rating_final.keys())
    model_order.sort(key=lambda k: -elo_rating_final[k])

    limit_show_number = 25  # limit show number to make plots smaller
    model_order = model_order[:limit_show_number]

    # leaderboard_table_df: elo rating, variance, 95% interval, number of battles
    leaderboard_table_df = pd.DataFrame(
        {
            "rating": elo_rating_final,
            "variance": bootstrap_df.var(),
            "rating_q975": bootstrap_df.quantile(0.975),
            "rating_q025": bootstrap_df.quantile(0.025),
            "num_battles": battles["model_a"].value_counts()
            + battles["model_b"].value_counts(),
        }
    )

    # Plots
    leaderboard_table = visualize_leaderboard_table(elo_rating_final)
    win_fraction_heatmap = visualize_pairwise_win_fraction(battles_no_ties, model_order)
    battle_count_heatmap = visualize_battle_count(battles_no_ties, model_order)
    average_win_rate_bar = visualize_average_win_rate(
        battles_no_ties, limit_show_number
    )
    bootstrap_elo_rating = visualize_bootstrap_elo_rating(
        bootstrap_df, elo_rating_final, limit_show_number
    )

    last_updated_tstamp = battles["tstamp"].max()
    last_updated_datetime = datetime.datetime.fromtimestamp(
        last_updated_tstamp, tz=timezone("US/Pacific")
    ).strftime("%Y-%m-%d %H:%M:%S %Z")

    return {
        "rating_system": rating_system,
        "elo_rating_online": elo_rating_online,
        "elo_rating_final": elo_rating_final,
        "leaderboard_table": leaderboard_table,
        "win_fraction_heatmap": win_fraction_heatmap,
        "battle_count_heatmap": battle_count_heatmap,
        "average_win_rate_bar": average_win_rate_bar,
        "bootstrap_elo_rating": bootstrap_elo_rating,
        "last_updated_datetime": last_updated_datetime,
        "last_updated_tstamp": last_updated_tstamp,
        "bootstrap_df": bootstrap_df,
        "leaderboard_table_df": leaderboard_table_df,
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
    parser.add_argument("--num-bootstrap", type=int, default=100)
    parser.add_argument(
        "--rating-system", type=str, choices=["bt", "elo"], default="bt"
    )
    parser.add_argument("--exclude-tie", action="store_true", default=False)
    args = parser.parse_args()

    np.random.seed(42)

    if args.clean_battle_file:
        # Read data from a cleaned battle files
        battles = pd.read_json(args.clean_battle_file)
    else:
        # Read data from all log files
        log_files = get_log_files(args.max_num_files)
        battles = clean_battle_data(log_files)

    results = report_elo_analysis_results(
        battles, rating_system=args.rating_system, num_bootstrap=args.num_bootstrap
    )

    print("# Online Elo")
    pretty_print_elo_rating(results["elo_rating_online"])
    print("# Median")
    pretty_print_elo_rating(results["elo_rating_final"])
    print(f"last update : {results['last_updated_datetime']}")

    last_updated_tstamp = results["last_updated_tstamp"]
    cutoff_date = datetime.datetime.fromtimestamp(
        last_updated_tstamp, tz=timezone("US/Pacific")
    ).strftime("%Y%m%d")

    with open(f"elo_results_{cutoff_date}.pkl", "wb") as fout:
        pickle.dump(results, fout)
