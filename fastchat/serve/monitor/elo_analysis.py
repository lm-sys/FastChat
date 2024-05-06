import argparse
import ast
from collections import defaultdict
import datetime
import json
import math
import pickle
from pytz import timezone
from functools import partial

import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm
from transformers import AutoTokenizer

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


def compute_elo_mle_with_tie(
    df, SCALE=400, BASE=10, INIT_RATING=1000, sample_weight=None
):
    from sklearn.linear_model import LogisticRegression

    ptbl_a_win = pd.pivot_table(
        df[df["winner"] == "model_a"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    ptbl_tie = pd.pivot_table(
        df[df["winner"].isin(["tie", "tie (bothbad)"])],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    ptbl_tie = ptbl_tie + ptbl_tie.T
    ptbl_b_win = pd.pivot_table(
        df[df["winner"] == "model_b"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    ptbl_win = ptbl_a_win * 2 + ptbl_b_win.T * 2 + ptbl_tie

    models = pd.Series(np.arange(len(ptbl_win.index)), index=ptbl_win.index)

    p = len(models)
    X = np.zeros([p * (p - 1) * 2, p])
    Y = np.zeros(p * (p - 1) * 2)

    cur_row = 0
    sample_weights = []
    for m_a in ptbl_win.index:
        for m_b in ptbl_win.columns:
            if m_a == m_b:
                continue
            # if nan skip
            if math.isnan(ptbl_win.loc[m_a, m_b]) or math.isnan(ptbl_win.loc[m_b, m_a]):
                continue
            X[cur_row, models[m_a]] = +math.log(BASE)
            X[cur_row, models[m_b]] = -math.log(BASE)
            Y[cur_row] = 1.0
            sample_weights.append(ptbl_win.loc[m_a, m_b])

            X[cur_row + 1, models[m_a]] = math.log(BASE)
            X[cur_row + 1, models[m_b]] = -math.log(BASE)
            Y[cur_row + 1] = 0.0
            sample_weights.append(ptbl_win.loc[m_b, m_a])
            cur_row += 2
    X = X[:cur_row]
    Y = Y[:cur_row]

    lr = LogisticRegression(fit_intercept=False, penalty=None)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = SCALE * lr.coef_[0] + INIT_RATING
    if "mixtral-8x7b-instruct-v0.1" in models.index:
        elo_scores += 1114 - elo_scores[models["mixtral-8x7b-instruct-v0.1"]]
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


def visualize_pairwise_win_fraction(battles, model_order, scale=1):
    row_beats_col = compute_pairwise_win_fraction(battles, model_order)
    fig = px.imshow(
        row_beats_col,
        color_continuous_scale="RdBu",
        text_auto=".2f",
        height=700 * scale,
        width=700 * scale,
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


def visualize_battle_count(battles, model_order, scale=1):
    ptbl = pd.pivot_table(
        battles, index="model_a", columns="model_b", aggfunc="size", fill_value=0
    )
    battle_counts = ptbl + ptbl.T
    fig = px.imshow(
        battle_counts.loc[model_order, model_order],
        text_auto=True,
        height=700 * scale,
        width=700 * scale,
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


def visualize_average_win_rate(battles, limit_show_number, scale=1):
    row_beats_col_freq = compute_pairwise_win_fraction(
        battles, None, limit_show_number=limit_show_number
    )
    fig = px.bar(
        row_beats_col_freq.mean(axis=1).sort_values(ascending=False),
        text_auto=".2f",
        height=500 * scale,
        width=700 * scale,
    )
    fig.update_layout(
        yaxis_title="Average Win Rate", xaxis_title="Model", showlegend=False
    )
    return fig


def visualize_bootstrap_elo_rating(df, df_final, limit_show_number, scale=1):
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
    bars["rating_rounded"] = np.round(bars["rating"])
    fig = px.scatter(
        bars,
        x="model",
        y="rating",
        error_y="error_y",
        error_y_minus="error_y_minus",
        text="rating_rounded",
        height=700,
        width=700 * scale,
    )
    fig.update_layout(xaxis_title="Model", yaxis_title="Rating")
    return fig


def limit_user_votes(battles, daily_vote_per_user):
    from datetime import datetime

    print("Before limiting user votes: ", len(battles))
    # add date
    battles["date"] = battles["tstamp"].apply(
        lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d")
    )

    battles_new = pd.DataFrame()
    for date in battles["date"].unique():
        # only take the first daily_vote_per_user votes per judge per day
        df_today = battles[battles["date"] == date]
        df_sub = df_today.groupby("judge").head(daily_vote_per_user)

        # add df_sub to a new dataframe
        battles_new = pd.concat([battles_new, df_sub])
    print("After limiting user votes: ", len(battles_new))
    return battles_new


def get_model_pair_stats(battles):
    battles["ordered_pair"] = battles.apply(
        lambda x: tuple(sorted([x["model_a"], x["model_b"]])), axis=1
    )

    model_pair_stats = {}

    for index, row in battles.iterrows():
        pair = row["ordered_pair"]
        if pair not in model_pair_stats:
            model_pair_stats[pair] = {"win": 0, "loss": 0, "tie": 0}

        if row["winner"] in ["tie", "tie (bothbad)"]:
            model_pair_stats[pair]["tie"] += 1
        elif row["winner"] == "model_a" and row["model_a"] == min(pair):
            model_pair_stats[pair]["win"] += 1
        elif row["winner"] == "model_b" and row["model_b"] == min(pair):
            model_pair_stats[pair]["win"] += 1
        else:
            model_pair_stats[pair]["loss"] += 1

    return model_pair_stats


def outlier_detect(
    model_pair_stats,
    battles,
    max_vote=100,
    randomized=False,
    alpha=0.05,
    c_param=0.5,
    user_list=None,
):
    if user_list is None:
        # only check user who has >= 5 votes to save compute
        user_vote_cnt = battles["judge"].value_counts()
        user_list = user_vote_cnt[user_vote_cnt >= 5].index.tolist()
    print("#User to be checked: ", len(user_list))

    bad_user_list = []
    for user in user_list:
        flag = False
        p_upper = []
        p_lower = []
        df_2 = battles[battles["judge"] == user]
        for row in df_2.iterrows():
            if len(p_upper) >= max_vote:
                break

            model_pair = tuple(sorted([row[1]["model_a"], row[1]["model_b"]]))

            if row[1]["winner"] in ["tie", "tie (bothbad)"]:
                vote = 0.5
            elif row[1]["winner"] == "model_a" and row[1]["model_a"] == model_pair[0]:
                vote = 1
            elif row[1]["winner"] == "model_b" and row[1]["model_b"] == model_pair[0]:
                vote = 1
            else:
                vote = 0

            stats = model_pair_stats[model_pair]
            # count all votes
            # ratings = np.array(
            #     [1] * stats["win"] + [0.5] * stats["tie"] + [0] * stats["loss"]
            # )

            # only count win and loss
            ratings = np.array([1] * stats["win"] + [0] * stats["loss"])
            if randomized:
                noise = np.random.uniform(-1e-5, 1e-5, len(ratings))
                ratings += noise
                vote += np.random.uniform(-1e-5, 1e-5)

            p_upper += [(ratings <= vote).mean()]
            p_lower += [(ratings >= vote).mean()]

            M_upper = np.prod(1 / (2 * np.array(p_upper)))
            M_lower = np.prod(1 / (2 * np.array(p_lower)))

            # M_upper = np.prod((1 - c_param) / (c_param * np.array(p_upper) ** c_param))
            # M_lower = np.prod((1 - c_param) / (c_param * np.array(p_lower) ** c_param))
            if (M_upper > 1 / alpha) or (M_lower > 1 / alpha):
                print(f"Identify bad user with {len(p_upper)} votes")
                flag = True
                break
        if flag:
            bad_user_list.append({"user_id": user, "votes": len(p_upper)})
    print("Bad user length: ", len(bad_user_list))
    print(bad_user_list)

    bad_user_id_list = [x["user_id"] for x in bad_user_list]
    # remove bad users
    battles = battles[~battles["judge"].isin(bad_user_id_list)]
    return battles


def filter_long_conv(row):
    threshold = 768
    for conversation_type in ["conversation_a", "conversation_b"]:
        cur_conv = row[conversation_type]
        num_tokens_all = sum([turn["num_tokens"] for turn in cur_conv])
        if num_tokens_all >= threshold:
            return True
    return False


def report_elo_analysis_results(
    battles_json,
    rating_system="bt",
    num_bootstrap=100,
    exclude_models=[],
    langs=[],
    exclude_tie=False,
    exclude_unknown_lang=False,
    daily_vote_per_user=None,
    run_outlier_detect=False,
    scale=1,
    filter_func=lambda x: True,
):
    battles = pd.DataFrame(battles_json)

    tqdm.pandas(desc=f"Processing using {filter_func.__name__}")
    filtered_indices = battles.progress_apply(filter_func, axis=1)
    battles = battles[filtered_indices]

    battles = battles.sort_values(ascending=True, by=["tstamp"])

    if len(langs) > 0:
        battles = battles[battles["language"].isin(langs)]
    if exclude_unknown_lang:
        battles = battles[~battles["language"].str.contains("unknown")]

    # remove excluded models
    battles = battles[
        ~(
            battles["model_a"].isin(exclude_models)
            | battles["model_b"].isin(exclude_models)
        )
    ]

    # Only use anonymous votes
    battles = battles[battles["anony"]].reset_index(drop=True)
    battles_no_ties = battles[~battles["winner"].str.contains("tie")]
    if exclude_tie:
        battles = battles_no_ties

    if daily_vote_per_user is not None:
        battles = limit_user_votes(battles, daily_vote_per_user)

    if run_outlier_detect:
        model_pair_stats = get_model_pair_stats(battles)
        battles = outlier_detect(model_pair_stats, battles)

    print(f"Number of battles: {len(battles)}")
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

    model_rating_q025 = bootstrap_df.quantile(0.025)
    model_rating_q975 = bootstrap_df.quantile(0.975)

    # compute ranking based on CI
    ranking = {}
    for i, model_a in enumerate(model_order):
        ranking[model_a] = 1
        for j, model_b in enumerate(model_order):
            if i == j:
                continue
            if model_rating_q025[model_b] > model_rating_q975[model_a]:
                ranking[model_a] += 1

    # leaderboard_table_df: elo rating, variance, 95% interval, number of battles
    leaderboard_table_df = pd.DataFrame(
        {
            "rating": elo_rating_final,
            "variance": bootstrap_df.var(),
            "rating_q975": bootstrap_df.quantile(0.975),
            "rating_q025": bootstrap_df.quantile(0.025),
            "num_battles": battles["model_a"]
            .value_counts()
            .add(battles["model_b"].value_counts(), fill_value=0),
            "final_ranking": pd.Series(ranking),
        }
    )

    model_order.sort(key=lambda k: -elo_rating_final[k])
    limit_show_number = int(25 * scale)
    model_order = model_order[:limit_show_number]

    # Plots
    leaderboard_table = visualize_leaderboard_table(elo_rating_final)
    win_fraction_heatmap = visualize_pairwise_win_fraction(
        battles_no_ties, model_order, scale=scale
    )
    battle_count_heatmap = visualize_battle_count(
        battles_no_ties, model_order, scale=scale
    )
    average_win_rate_bar = visualize_average_win_rate(
        battles_no_ties, limit_show_number, scale=scale
    )
    bootstrap_elo_rating = visualize_bootstrap_elo_rating(
        bootstrap_df, elo_rating_final, limit_show_number, scale=scale
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
    parser.add_argument("--exclude-models", type=str, nargs="+", default=[])
    parser.add_argument("--exclude-tie", action="store_true", default=False)
    parser.add_argument("--exclude-unknown-lang", action="store_true", default=False)
    parser.add_argument("--exclude-url", action="store_true", default=False)
    parser.add_argument("--langs", type=str, nargs="+", default=[])
    parser.add_argument("--daily-vote-per-user", type=int, default=None)
    parser.add_argument("--run-outlier-detect", action="store_true", default=False)
    parser.add_argument("--category", nargs="+", default=["full"])
    parser.add_argument("--scale", type=float, default=1)
    args = parser.parse_args()

    np.random.seed(42)

    if args.clean_battle_file:
        # Read data from a cleaned battle files
        battles = pd.read_json(args.clean_battle_file)
    else:
        # Read data from all log files
        log_files = get_log_files(args.max_num_files)
        battles = clean_battle_data(log_files)

    filter_func_map = {
        "full": lambda x: True,
        "long": filter_long_conv,
        "chinese": lambda x: x["language"] == "Chinese",
        "english": lambda x: x["language"] == "English",
    }
    assert all(
        [cat in filter_func_map for cat in args.category]
    ), f"Invalid category: {args.category}"

    results = {}
    for cat in args.category:
        filter_func = filter_func_map[cat]
        results[cat] = report_elo_analysis_results(
            battles,
            rating_system=args.rating_system,
            num_bootstrap=args.num_bootstrap,
            exclude_models=args.exclude_models,
            langs=args.langs,
            exclude_tie=args.exclude_tie,
            exclude_unknown_lang=args.exclude_unknown_lang,
            daily_vote_per_user=args.daily_vote_per_user,
            run_outlier_detect=args.run_outlier_detect,
            scale=args.scale,
            filter_func=filter_func,
        )

    for cat in args.category:
        print(f"# Results for {cat} conversations")
        print("# Online Elo")
        pretty_print_elo_rating(results[cat]["elo_rating_online"])
        print("# Median")
        pretty_print_elo_rating(results[cat]["elo_rating_final"])
        print(f"last update : {results[cat]['last_updated_datetime']}")

        last_updated_tstamp = results[cat]["last_updated_tstamp"]
        cutoff_date = datetime.datetime.fromtimestamp(
            last_updated_tstamp, tz=timezone("US/Pacific")
        ).strftime("%Y%m%d")
        print(f"last update : {cutoff_date}")

    with open(f"elo_results_{cutoff_date}.pkl", "wb") as fout:
        pickle.dump(results, fout)
