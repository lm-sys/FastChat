import argparse
from collections import defaultdict
import json
import math

import gdown
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm

from fastchat.serve.monitor.basic_stats import get_log_files
from fastchat.serve.monitor.clean_battle_data import clean_battle_data


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
    
    return rating


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
                    text_auto=".2f")
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
    fig = px.imshow(battle_counts.loc[model_order, model_order], text_auto=True)
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
                 title="Average Win Rate Against All Other Models (Assuming Uniform Sampling and No Ties)", 
                 text_auto=".2f")
    fig.update_layout(yaxis_title="Average Win Rate", xaxis_title="Model",
                      showlegend=False)
    return fig


def report_elo_analysis_results(log_files):
    battles = clean_battle_data(log_files)
    battles = pd.DataFrame(battles)
    # Only use anonymous votes
    battles = battles[battles["anony"]].reset_index(drop=True)
    battles_no_ties = battles[~battles["win"].str.contains("tie")]

    elo_rating = compute_elo(battles)

    model_order = list(elo_rating.keys())
    model_order.sort(key=lambda k: -elo_rating[k])

    win_fraction_heatmap = visualize_pairwise_win_fraction(battles_no_ties, model_order)
    battle_count_heatmap = visualize_battle_count(battles_no_ties, model_order)
    average_win_rate_bar = visualize_average_win_rate(battles_no_ties)

    return {
        "elo_rating": elo_rating,
        "win_fraction_heatmap": win_fraction_heatmap,
        "battle_count_heatmap": battle_count_heatmap,
        "average_win_rate_bar": average_win_rate_bar,
    }


def pretty_print_elo_rating(rating):
    model_order = list(rating.keys())
    model_order.sort(key=lambda k: -rating[k])
    for i, model in enumerate(model_order):
        print(f"{i+1:2d}, {model:25s}, {rating[model]:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-num-files", type=int)
    args = parser.parse_args()

    log_files = get_log_files(args.max_num_files)
    results = report_elo_analysis_results(log_files)

    print()
    pretty_print_elo_rating(results["elo_rating"])
