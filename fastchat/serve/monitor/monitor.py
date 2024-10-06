"""
Live monitor of the website statistics and leaderboard.

Dependency:
sudo apt install pkg-config libicu-dev
pip install pytz gradio gdown plotly polyglot pyicu pycld2 tabulate
"""

import argparse
import ast
import json
import pickle
import os
import threading
import time

import pandas as pd
import gradio as gr
import numpy as np

from fastchat.constants import SURVEY_LINK
from fastchat.serve.monitor.basic_stats import report_basic_stats, get_log_files
from fastchat.serve.monitor.clean_battle_data import clean_battle_data
from fastchat.serve.monitor.elo_analysis import report_elo_analysis_results
from fastchat.utils import build_logger, get_window_url_params_js


from fastchat.serve.monitor.monitor_md import (
    cat_name_to_baseline,
    key_to_category_name,
    cat_name_to_explanation,
    deprecated_model_name,
    arena_hard_title,
    make_default_md_1,
    make_default_md_2,
    make_arena_leaderboard_md,
    make_category_arena_leaderboard_md,
    make_full_leaderboard_md,
    make_leaderboard_md_live,
)

k2c = {}
for k, v in key_to_category_name.items():
    k2c[k] = v
    k2c[k + "_style_control"] = v + "_style_control"
key_to_category_name = k2c

notebook_url = (
    "https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH"
)

basic_component_values = [None] * 6
leader_component_values = [None] * 5


def recompute_final_ranking(arena_df):
    # compute ranking based on CI
    ranking = {}
    for i, model_a in enumerate(arena_df.index):
        ranking[model_a] = 1
        for j, model_b in enumerate(arena_df.index):
            if i == j:
                continue
            if (
                arena_df.loc[model_b]["rating_q025"]
                > arena_df.loc[model_a]["rating_q975"]
            ):
                ranking[model_a] += 1
    return list(ranking.values())


def arena_hard_title(date):
    arena_hard_title = f"""
Last Updated: {date}

**Arena-Hard-Auto v0.1** - an automatic evaluation tool for instruction-tuned LLMs with 500 challenging user queries curated from Chatbot Arena. 

We prompt GPT-4-Turbo as judge to compare the models' responses against a baseline model (default: GPT-4-0314). If you are curious to see how well your model might perform on Chatbot Arena, we recommend trying Arena-Hard-Auto. Check out our paper for more details about how Arena-Hard-Auto works as an fully automated data pipeline converting crowdsourced data into high-quality benchmarks ->
[[Paper](https://arxiv.org/abs/2406.11939) | [Repo](https://github.com/lm-sys/arena-hard-auto)]
    """
    return arena_hard_title


def recompute_final_ranking(arena_df):
    # compute ranking based on CI
    ranking = {}
    for i, model_a in enumerate(arena_df.index):
        ranking[model_a] = 1
        for j, model_b in enumerate(arena_df.index):
            if i == j:
                continue
            if (
                arena_df.loc[model_b]["rating_q025"]
                > arena_df.loc[model_a]["rating_q975"]
            ):
                ranking[model_a] += 1
    return list(ranking.values())


def update_elo_components(
    max_num_files, elo_results_file, ban_ip_file, exclude_model_names
):
    log_files = get_log_files(max_num_files)

    # Leaderboard
    if elo_results_file is None:  # Do live update
        ban_ip_list = json.load(open(ban_ip_file)) if ban_ip_file else None
        battles = clean_battle_data(
            log_files, exclude_model_names, ban_ip_list=ban_ip_list
        )
        elo_results = report_elo_analysis_results(battles, scale=2)

        leader_component_values[0] = make_leaderboard_md_live(elo_results)
        leader_component_values[1] = elo_results["win_fraction_heatmap"]
        leader_component_values[2] = elo_results["battle_count_heatmap"]
        leader_component_values[3] = elo_results["bootstrap_elo_rating"]
        leader_component_values[4] = elo_results["average_win_rate_bar"]

    # Basic stats
    basic_stats = report_basic_stats(log_files)
    md0 = f"Last updated: {basic_stats['last_updated_datetime']}"

    md1 = "### Action Histogram\n"
    md1 += basic_stats["action_hist_md"] + "\n"

    md2 = "### Anony. Vote Histogram\n"
    md2 += basic_stats["anony_vote_hist_md"] + "\n"

    md3 = "### Model Call Histogram\n"
    md3 += basic_stats["model_hist_md"] + "\n"

    md4 = "### Model Call (Last 24 Hours)\n"
    md4 += basic_stats["num_chats_last_24_hours"] + "\n"

    basic_component_values[0] = md0
    basic_component_values[1] = basic_stats["chat_dates_bar"]
    basic_component_values[2] = md1
    basic_component_values[3] = md2
    basic_component_values[4] = md3
    basic_component_values[5] = md4


def update_worker(
    max_num_files, interval, elo_results_file, ban_ip_file, exclude_model_names
):
    while True:
        tic = time.time()
        update_elo_components(
            max_num_files, elo_results_file, ban_ip_file, exclude_model_names
        )
        durtaion = time.time() - tic
        print(f"update duration: {durtaion:.2f} s")
        time.sleep(max(interval - durtaion, 0))


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")
    return basic_component_values + leader_component_values


def model_hyperlink(model_name, link):
    return f'<a target="_blank" href="{link}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{model_name}</a>'


def load_leaderboard_table_csv(filename, add_hyperlink=True):
    lines = open(filename).readlines()
    heads = [v.strip() for v in lines[0].split(",")]
    rows = []
    for i in range(1, len(lines)):
        row = [v.strip() for v in lines[i].split(",")]
        for j in range(len(heads)):
            item = {}
            for h, v in zip(heads, row):
                if h == "Arena Elo rating":
                    if v != "-":
                        v = int(ast.literal_eval(v))
                    else:
                        v = np.nan
                elif h == "MMLU":
                    if v != "-":
                        v = round(ast.literal_eval(v) * 100, 1)
                    else:
                        v = np.nan
                elif h == "MT-bench (win rate %)":
                    if v != "-":
                        v = round(ast.literal_eval(v[:-1]), 1)
                    else:
                        v = np.nan
                elif h == "MT-bench (score)":
                    if v != "-":
                        v = round(ast.literal_eval(v), 2)
                    else:
                        v = np.nan
                item[h] = v
            if add_hyperlink:
                item["Model"] = model_hyperlink(item["Model"], item["Link"])
        rows.append(item)

    return rows


def build_basic_stats_tab():
    empty = "Loading ..."
    basic_component_values[:] = [empty, None, empty, empty, empty, empty]

    md0 = gr.Markdown(empty)
    gr.Markdown("#### Figure 1: Number of model calls and votes")
    plot_1 = gr.Plot(show_label=False)
    with gr.Row():
        with gr.Column():
            md1 = gr.Markdown(empty)
        with gr.Column():
            md2 = gr.Markdown(empty)
    with gr.Row():
        with gr.Column():
            md3 = gr.Markdown(empty)
        with gr.Column():
            md4 = gr.Markdown(empty)
    return [md0, plot_1, md1, md2, md3, md4]


def get_full_table(arena_df, model_table_df, model_to_score):
    values = []
    for i in range(len(model_table_df)):
        row = []
        model_key = model_table_df.iloc[i]["key"]
        model_name = model_table_df.iloc[i]["Model"]
        # model display name
        row.append(model_name)
        if model_key in arena_df.index:
            idx = arena_df.index.get_loc(model_key)
            row.append(round(arena_df.iloc[idx]["rating"]))
        else:
            row.append(np.nan)
        if model_name in model_to_score:
            row.append(model_to_score[model_name])
        else:
            row.append(np.nan)
        row.append(model_table_df.iloc[i]["MT-bench (score)"])
        row.append(model_table_df.iloc[i]["MMLU"])
        # Organization
        row.append(model_table_df.iloc[i]["Organization"])
        # license
        row.append(model_table_df.iloc[i]["License"])

        values.append(row)
    values.sort(key=lambda x: -x[1] if not np.isnan(x[1]) else 1e9)
    return values


def arena_hard_process(leaderboard_table_file, filepath):
    arena_hard = pd.read_csv(filepath)
    leaderboard_table = pd.read_csv(leaderboard_table_file)
    links = leaderboard_table.get("Link")
    display_name = leaderboard_table.get("Model")
    model_name = leaderboard_table.get("key")
    organization = leaderboard_table.get("Organization")

    info = {}
    for i in range(len(model_name)):
        model_info = {}
        model_info["display"] = display_name[i]
        model_info["link"] = links[i]
        model_info["org"] = organization[i]
        info[model_name[i]] = model_info

    organization = []
    for i in range(len(arena_hard)):
        assert (
            arena_hard.loc[i, "model"] in info
        ), f"need to update leaderboard_table info by adding {arena_hard.loc[i, 'model']}"
        organization.append(info[arena_hard.loc[i, "model"]]["org"])
        link = info[arena_hard.loc[i, "model"]]["link"]
        arena_hard.loc[i, "model"] = model_hyperlink(
            info[arena_hard.loc[i, "model"]]["display"], link
        )

    arena_hard.insert(
        loc=len(arena_hard.columns), column="Organization", value=organization
    )

    rankings = recompute_final_ranking(arena_hard)
    arena_hard.insert(loc=0, column="Rank* (UB)", value=rankings)
    return arena_hard


def create_ranking_str(ranking, ranking_difference):
    if ranking_difference > 0:
        return f"{int(ranking)} \u2191"
    elif ranking_difference < 0:
        return f"{int(ranking)} \u2193"
    else:
        return f"{int(ranking)}"


def get_arena_table(arena_df, model_table_df, arena_subset_df=None, hidden_models=None):
    arena_df = arena_df.sort_values(
        by=["final_ranking", "rating"], ascending=[True, False]
    )

    if hidden_models:
        arena_df = arena_df[~arena_df.index.isin(hidden_models)].copy()

    arena_df["final_ranking"] = recompute_final_ranking(arena_df)

    if arena_subset_df is not None:
        arena_subset_df = arena_subset_df[arena_subset_df.index.isin(arena_df.index)]
        arena_subset_df = arena_subset_df.sort_values(by=["rating"], ascending=False)
        arena_subset_df["final_ranking"] = recompute_final_ranking(arena_subset_df)

        arena_df = arena_df[arena_df.index.isin(arena_subset_df.index)]
        arena_df["final_ranking"] = recompute_final_ranking(arena_df)

        arena_subset_df["final_ranking_no_tie"] = np.arange(1, len(arena_subset_df) + 1)
        arena_df["final_ranking_no_tie"] = np.arange(1, len(arena_df) + 1)

        arena_df = arena_subset_df.join(
            arena_df["final_ranking"], rsuffix="_global", how="inner"
        )
        arena_df["ranking_difference"] = (
            arena_df["final_ranking_global"] - arena_df["final_ranking"]
        )

        arena_df = arena_df.sort_values(
            by=["final_ranking", "rating"], ascending=[True, False]
        )
        arena_df["final_ranking"] = arena_df.apply(
            lambda x: create_ranking_str(x["final_ranking"], x["ranking_difference"]),
            axis=1,
        )

    arena_df["final_ranking"] = arena_df["final_ranking"].astype(str)

    # Handle potential duplicate keys in model_table_df
    model_table_dict = model_table_df.groupby("key").first().to_dict(orient="index")

    def process_row(row):
        model_key = row.name
        model_info = model_table_dict.get(model_key, {})

        if not model_info:
            print(f"Warning: {model_key} not found in model table")
            return None

        ranking = row.get("final_ranking") or row.name + 1
        result = [ranking]

        if arena_subset_df is not None:
            result.append(row.get("ranking_difference", 0))

        result.extend(
            [
                model_info.get("Model", "Unknown"),
                f"{round(row['rating'])}",
                f"+{round(row['rating_q975'] - row['rating'])}/-{round(row['rating'] - row['rating_q025'])}",
                round(row["num_battles"]),
                model_info.get("Organization", "Unknown"),
                model_info.get("License", "Unknown"),
                (
                    "Unknown"
                    if model_info.get("Knowledge cutoff date", "-") == "-"
                    else model_info.get("Knowledge cutoff date", "Unknown")
                ),
            ]
        )

        return result

    values = [
        process_row(row)
        for _, row in arena_df.iterrows()
        if process_row(row) is not None
    ]

    return values


def update_leaderboard_df(arena_table_vals):
    columns = [
        "Rank* (UB)",
        "Delta",
        "Model",
        "Arena Score",
        "95% CI",
        "Votes",
        "Organization",
        "License",
        "Knowledge Cutoff",
    ]
    elo_dataframe = pd.DataFrame(arena_table_vals, columns=columns)

    def highlight_max(s):
        return [
            (
                "color: green; font-weight: bold"
                if "\u2191" in str(v)
                else "color: red; font-weight: bold"
                if "\u2193" in str(v)
                else ""
            )
            for v in s
        ]

    def highlight_rank_max(s):
        return [
            (
                "color: green; font-weight: bold"
                if v > 0
                else "color: red; font-weight: bold"
                if v < 0
                else ""
            )
            for v in s
        ]

    return elo_dataframe.style.apply(highlight_max, subset=["Rank* (UB)"]).apply(
        highlight_rank_max, subset=["Delta"]
    )


def build_arena_tab(
    elo_results,
    model_table_df,
    default_md,
    vision=False,
    show_plot=False,
):
    if elo_results is None:
        gr.Markdown(
            """ ## Coming soon...!
            """,
        )
        return

    arena_dfs = {}
    category_elo_results = {}
    last_updated_time = elo_results["full"]["last_updated_datetime"].split(" ")[0]

    for k in key_to_category_name.keys():
        if k not in elo_results:
            continue
        arena_dfs[key_to_category_name[k]] = elo_results[k]["leaderboard_table_df"]
        category_elo_results[key_to_category_name[k]] = elo_results[k]

    arena_df = arena_dfs["Overall"]

    def update_leaderboard_and_plots(category, filters):
        if len(filters) > 0 and "Style Control" in filters:
            cat_name = f"{category} w/ Style Control"
            if cat_name in arena_dfs:
                category = cat_name
            else:
                gr.Warning("This category does not support style control.")

        arena_subset_df = arena_dfs[category]
        arena_subset_df = arena_subset_df[arena_subset_df["num_battles"] > 300]
        elo_subset_results = category_elo_results[category]

        baseline_category = cat_name_to_baseline.get(category, "Overall")
        arena_df = arena_dfs[baseline_category]
        arena_values = get_arena_table(
            arena_df,
            model_table_df,
            arena_subset_df=arena_subset_df if category != "Overall" else None,
            hidden_models=(
                None
                if len(filters) > 0 and "Show Deprecate" in filters
                else deprecated_model_name
            ),
        )
        if category != "Overall":
            arena_values = update_leaderboard_df(arena_values)
            # arena_values = highlight_top_models(arena_values)
            arena_values = gr.Dataframe(
                headers=[
                    "Rank* (UB)",
                    "Delta",
                    "Model",
                    "Arena Score",
                    "95% CI",
                    "Votes",
                    "Organization",
                    "License",
                    "Knowledge Cutoff",
                ],
                datatype=[
                    "str",
                    "number",
                    "markdown",
                    "number",
                    "str",
                    "number",
                    "str",
                    "str",
                    "str",
                ],
                value=arena_values,
                elem_id="arena_leaderboard_dataframe",
                height=1000,
                column_widths=[70, 70, 210, 90, 90, 90, 120, 150, 100],
                wrap=True,
            )
        else:
            arena_values = gr.Dataframe(
                headers=[
                    "Rank* (UB)",
                    "Model",
                    "Arena Score",
                    "95% CI",
                    "Votes",
                    "Organization",
                    "License",
                    "Knowledge Cutoff",
                ],
                datatype=[
                    "number",
                    "markdown",
                    "number",
                    "str",
                    "number",
                    "str",
                    "str",
                    "str",
                ],
                value=arena_values,
                elem_id="arena_leaderboard_dataframe",
                height=1000,
                column_widths=[70, 220, 90, 90, 90, 120, 150, 100],
                wrap=True,
            )

        p1 = elo_subset_results["win_fraction_heatmap"]
        p2 = elo_subset_results["battle_count_heatmap"]
        p3 = elo_subset_results["bootstrap_elo_rating"]
        p4 = elo_subset_results["average_win_rate_bar"]
        more_stats_md = f"""## More Statistics for Chatbot Arena - {category}
        """
        leaderboard_md = make_category_arena_leaderboard_md(
            arena_df, arena_subset_df, name=category
        )
        return arena_values, p1, p2, p3, p4, more_stats_md, leaderboard_md

    arena_df = arena_dfs["Overall"]

    p1 = category_elo_results["Overall"]["win_fraction_heatmap"]
    p2 = category_elo_results["Overall"]["battle_count_heatmap"]
    p3 = category_elo_results["Overall"]["bootstrap_elo_rating"]
    p4 = category_elo_results["Overall"]["average_win_rate_bar"]

    # arena table
    arena_table_vals = get_arena_table(
        arena_df, model_table_df, hidden_models=deprecated_model_name
    )

    md = make_arena_leaderboard_md(arena_df, last_updated_time, vision=vision)
    gr.Markdown(md, elem_id="leaderboard_markdown")

    # only keep category without style control
    category_choices = list(arena_dfs.keys())
    category_choices = [x for x in category_choices if "Style Control" not in x]

    with gr.Row():
        with gr.Column(scale=2):
            category_dropdown = gr.Dropdown(
                choices=category_choices,
                label="Category",
                value="Overall",
            )
        with gr.Column(scale=2):
            category_checkbox = gr.CheckboxGroup(
                ["Style Control", "Show Deprecate"], label="Apply filter", info=""
            )
        default_category_details = make_category_arena_leaderboard_md(
            arena_df, arena_df, name="Overall"
        )
        with gr.Column(scale=4, variant="panel"):
            category_deets = gr.Markdown(
                default_category_details, elem_id="category_deets"
            )

    arena_vals = pd.DataFrame(
        arena_table_vals,
        columns=[
            "Rank* (UB)",
            "Model",
            "Arena Score",
            "95% CI",
            "Votes",
            "Organization",
            "License",
            "Knowledge Cutoff",
        ],
    )
    elo_display_df = gr.Dataframe(
        headers=[
            "Rank* (UB)",
            "Model",
            "Arena Elo",
            "95% CI",
            "Votes",
            "Organization",
            "License",
            "Knowledge Cutoff",
        ],
        datatype=[
            "number",
            "markdown",
            "number",
            "str",
            "number",
            "str",
            "str",
            "str",
        ],
        # value=highlight_top_models(arena_vals.style),
        value=arena_vals.style,
        elem_id="arena_leaderboard_dataframe",
        height=1000,
        column_widths=[70, 220, 90, 90, 90, 120, 150, 100],
        wrap=True,
    )

    gr.Markdown(
        f"""
***Rank (UB)**: model's ranking (upper-bound), defined by one + the number of models that are statistically better than the target model.
Model A is statistically better than model B when A's lower-bound score is greater than B's upper-bound score (in 95% confidence interval).
See Figure 1 below for visualization of the confidence intervals of model scores.

Note: in each category, we exclude models with fewer than 300 votes as their confidence intervals can be large.
""",
        elem_id="leaderboard_markdown",
    )

    if not vision:
        leader_component_values[:] = [default_md, p1, p2, p3, p4]

    if show_plot:
        more_stats_md = gr.Markdown(
            f"""## More Statistics for Chatbot Arena (Overall)""",
            elem_id="leaderboard_header_markdown",
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    "#### Figure 1: Confidence Intervals on Model Strength (via Bootstrapping)",
                    elem_id="plot-title",
                )
                plot_3 = gr.Plot(p3, show_label=False)
            with gr.Column():
                gr.Markdown(
                    "#### Figure 2: Average Win Rate Against All Other Models (Assuming Uniform Sampling and No Ties)",
                    elem_id="plot-title",
                )
                plot_4 = gr.Plot(p4, show_label=False)
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    "#### Figure 3: Fraction of Model A Wins for All Non-tied A vs. B Battles",
                    elem_id="plot-title",
                )
                plot_1 = gr.Plot(p1, show_label=False, elem_id="plot-container")
            with gr.Column():
                gr.Markdown(
                    "#### Figure 4: Battle Count for Each Combination of Models (without Ties)",
                    elem_id="plot-title",
                )
                plot_2 = gr.Plot(p2, show_label=False)
    category_dropdown.change(
        update_leaderboard_and_plots,
        inputs=[category_dropdown, category_checkbox],
        outputs=[
            elo_display_df,
            plot_1,
            plot_2,
            plot_3,
            plot_4,
            more_stats_md,
            category_deets,
        ],
    )

    category_checkbox.change(
        update_leaderboard_and_plots,
        inputs=[category_dropdown, category_checkbox],
        outputs=[
            elo_display_df,
            plot_1,
            plot_2,
            plot_3,
            plot_4,
            more_stats_md,
            category_deets,
        ],
    )
    return [plot_1, plot_2, plot_3, plot_4]


def build_full_leaderboard_tab(elo_results, model_table_df, model_to_score):
    arena_df = elo_results["full"]["leaderboard_table_df"]
    md = make_full_leaderboard_md()
    gr.Markdown(md, elem_id="leaderboard_markdown")
    full_table_vals = get_full_table(arena_df, model_table_df, model_to_score)
    gr.Dataframe(
        headers=[
            "Model",
            "Arena Score",
            "arena-hard-auto",
            "MT-bench",
            "MMLU",
            "Organization",
            "License",
        ],
        datatype=["markdown", "number", "number", "number", "number", "str", "str"],
        value=full_table_vals,
        elem_id="full_leaderboard_dataframe",
        column_widths=[200, 100, 110, 100, 70, 130, 150],
        height=1000,
        wrap=True,
    )


def get_arena_category_table(results_df, categories, metric="ranking"):
    assert metric in ["rating", "ranking"]

    category_names = [key_to_category_name[k] for k in categories]
    filtered_df = results_df[results_df["category"].isin(category_names)][
        ["category", metric]
    ]
    category_df = filtered_df.pivot(columns="category", values=metric)
    category_df = category_df.fillna(-1).astype(int)

    # Reorder columns to match the input order of categories
    category_df = category_df.reindex(columns=category_names)
    category_df.insert(0, "Model", category_df.index)

    # insert model rating as a column to category_df
    category_df = category_df.merge(
        results_df[results_df["category"] == "Overall"][["Model", "rating"]],
        on="Model",
        how="left",
    )
    category_df = category_df.sort_values(
        by=[category_names[0], "rating"],
        ascending=[metric == "ranking", False],
    )
    # by=["final_ranking", "rating"], ascending=[True, False]
    category_df = category_df.drop(columns=["rating"])
    category_df = category_df.reset_index(drop=True)

    style = category_df.style

    def highlight_top_3(s):
        return [
            (
                "background-color: rgba(255, 215, 0, 0.5); text-align: center; font-size: 110%"
                if v == 1 and v != 0
                else (
                    "background-color: rgba(192, 192, 192, 0.5); text-align: center; font-size: 110%"
                    if v == 2 and v != 0
                    else (
                        "background-color: rgba(255, 165, 0, 0.5); text-align: center; font-size: 110%"
                        if v == 3 and v != 0
                        else "text-align: center; font-size: 110%"
                    )
                )
            )
            for v in s
        ]

    # Apply styling for each category
    for category in category_names:
        style = style.apply(highlight_top_3, subset=[category])

    if metric == "rating":
        style = style.background_gradient(
            cmap="Blues",
            subset=category_names,
            vmin=1150,
            vmax=category_df[category_names].max().max(),
        )

    return style


def build_category_leaderboard_tab(
    combined_elo_df, title, categories, categories_width
):
    full_table_vals = get_arena_category_table(combined_elo_df, categories)
    ranking_table_vals = get_arena_category_table(combined_elo_df, categories)
    rating_table_vals = get_arena_category_table(combined_elo_df, categories, "rating")
    with gr.Row():
        gr.Markdown(
            f"""&emsp; <span style='font-weight: bold; font-size: 125%;'>{title} Leaderboard</span>"""
        )
        ranking_button = gr.Button("Sort by Rank")
        rating_button = gr.Button("Sort by Arena Score")
        sort_rating = lambda _: get_arena_category_table(
            combined_elo_df, categories, "rating"
        )
        sort_ranking = lambda _: get_arena_category_table(combined_elo_df, categories)
    with gr.Row():
        gr.Markdown(
            f"""&emsp; <span style='font-weight: bold; font-size: 150%;'>Chatbot Arena Overview</span>"""
        )

    overall_ranking_leaderboard = gr.Dataframe(
        headers=["Model"] + [key_to_category_name[k] for k in categories],
        datatype=["markdown"] + ["str" for k in categories],
        value=full_table_vals,
        elem_id="full_leaderboard_dataframe",
        column_widths=[150]
        + categories_width,  # IMPORTANT: THIS IS HARDCODED WITH THE CURRENT CATEGORIES
        height=1000,
        wrap=True,
    )
    ranking_button.click(
        sort_ranking, inputs=[ranking_button], outputs=[overall_ranking_leaderboard]
    )
    rating_button.click(
        sort_rating, inputs=[rating_button], outputs=[overall_ranking_leaderboard]
    )


selected_categories = [
    "full",
    "full_style_control",
    "hard_6",
    "hard_6_style_control",
    "if",
    "coding",
    "math",
    "multiturn",
    "long_user",
    # "no_refusal",
]
# selected_categories_width = [95, 85, 100, 75, 120, 100, 95, 100,100]
selected_categories_width = [110, 110, 110, 110, 110, 80, 80, 80, 80]
# selected_categories_width = [100] * len(selected_categories)

language_categories = [
    "english",
    "chinese",
    "german",
    "french",
    "spanish",
    "russian",
    "japanese",
    "korean",
]
language_categories_width = [100] * len(language_categories)


def get_combined_table(elo_results, model_table_df):
    def get_model_name(model_key):
        try:
            model_name = model_table_df[model_table_df["key"] == model_key][
                "Model"
            ].values[0]
            return model_name
        except:
            return None

    combined_table = []
    for category in elo_results.keys():
        df = elo_results[category]["leaderboard_table_df"]
        ranking = recompute_final_ranking(df)
        df["ranking"] = ranking
        df["category"] = key_to_category_name[category]
        df["Model"] = df.index
        try:
            df["Model"] = df["Model"].apply(get_model_name)
            combined_table.append(df)
        except Exception as e:
            print(f"Error: {e}")
            continue
    combined_table = pd.concat(combined_table)
    combined_table["Model"] = combined_table.index
    # drop any rows with nan values
    combined_table = combined_table.dropna()
    return combined_table


def build_leaderboard_tab(
    elo_results_file,
    leaderboard_table_file,
    arena_hard_leaderboard,
    show_plot=False,
    mirror=False,
):
    if elo_results_file is None:  # Do live update
        default_md = "Loading ..."
        p1 = p2 = p3 = p4 = None
    else:
        with open(elo_results_file, "rb") as fin:
            elo_results = pickle.load(fin)
        if "text" in elo_results:
            elo_results_text = elo_results["text"]
            elo_results_vision = elo_results["vision"]
        else:
            elo_results_text = elo_results
            elo_results_vision = None

    default_md = make_default_md_1(mirror=mirror)
    default_md_2 = make_default_md_2(mirror=mirror)

    with gr.Row():
        with gr.Column(scale=4):
            md_1 = gr.Markdown(default_md, elem_id="leaderboard_markdown")
        if mirror:
            with gr.Column(scale=1):
                vote_button = gr.Button("Vote!", link="https://lmarena.ai")
    md2 = gr.Markdown(default_md_2, elem_id="leaderboard_markdown")
    if leaderboard_table_file:
        data = load_leaderboard_table_csv(leaderboard_table_file)
        model_table_df = pd.DataFrame(data)

        with gr.Tabs() as tabs:
            with gr.Tab("Arena", id=0):
                gr_plots = build_arena_tab(
                    elo_results_text,
                    model_table_df,
                    default_md,
                    show_plot=show_plot,
                )
            with gr.Tab("📣 NEW: Overview", id=1):
                gr.Markdown(
                    f"""
                    <div style="text-align: center; font-weight: bold;">
                        For a more holistic comparison, we've updated the leaderboard to show model rank (UB) across tasks and languages. Check out the 'Arena' tab for more categories, statistics, and model info.
                    </div>
                    """,
                )
                last_updated_time = elo_results_text["full"][
                    "last_updated_datetime"
                ].split(" ")[0]
                gr.Markdown(
                    make_arena_leaderboard_md(
                        elo_results_text["full"]["leaderboard_table_df"],
                        last_updated_time,
                    ),
                    elem_id="leaderboard_markdown",
                )
                combined_table = get_combined_table(elo_results_text, model_table_df)
                build_category_leaderboard_tab(
                    combined_table,
                    "Task",
                    selected_categories,
                    selected_categories_width,
                )
                build_category_leaderboard_tab(
                    combined_table,
                    "Language",
                    language_categories,
                    language_categories_width,
                )
                gr.Markdown(
                    f"""
            ***Rank (UB)**: model's ranking (upper-bound), defined by one + the number of models that are statistically better than the target model.
            Model A is statistically better than model B when A's lower-bound score is greater than B's upper-bound score (in 95% confidence interval).
            See Figure 1 below for visualization of the confidence intervals of model scores.

            Note: in each category, we exclude models with fewer than 300 votes as their confidence intervals can be large.
            """,
                    elem_id="leaderboard_markdown",
                )
            with gr.Tab("Arena (Vision)", id=2):
                build_arena_tab(
                    elo_results_vision,
                    model_table_df,
                    default_md,
                    vision=True,
                    show_plot=show_plot,
                )
            model_to_score = {}
            if arena_hard_leaderboard is not None:
                with gr.Tab("Arena-Hard-Auto", id=3):
                    dataFrame = arena_hard_process(
                        leaderboard_table_file, arena_hard_leaderboard
                    )
                    date = dataFrame["date"][0]
                    dataFrame = dataFrame.drop(
                        columns=["rating_q025", "rating_q975", "date"]
                    )
                    dataFrame["CI"] = dataFrame.CI.map(ast.literal_eval)
                    dataFrame["CI"] = dataFrame.CI.map(lambda x: f"+{x[1]}/-{x[0]}")
                    dataFrame = dataFrame.rename(
                        columns={
                            "model": "Model",
                            "score": "Win-rate",
                            "CI": "95% CI",
                            "avg_tokens": "Average Tokens",
                        }
                    )
                    model_to_score = {}
                    for i in range(len(dataFrame)):
                        model_to_score[dataFrame.loc[i, "Model"]] = dataFrame.loc[
                            i, "Win-rate"
                        ]
                    md = arena_hard_title(date)
                    gr.Markdown(md, elem_id="leaderboard_markdown")
                    gr.DataFrame(
                        dataFrame,
                        datatype=[
                            "markdown" if col == "Model" else "str"
                            for col in dataFrame.columns
                        ],
                        elem_id="arena_hard_leaderboard",
                        height=1000,
                        wrap=True,
                        column_widths=[70, 190, 80, 80, 90, 150],
                    )

            with gr.Tab("Full Leaderboard", id=4):
                build_full_leaderboard_tab(
                    elo_results_text, model_table_df, model_to_score
                )

        if not show_plot:
            gr.Markdown(
                """ ## Visit our [HF space](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) for more analysis!
                If you want to see more models, please help us [add them](https://github.com/lm-sys/FastChat/blob/main/docs/arena.md#how-to-add-a-new-model).
                """,
                elem_id="leaderboard_markdown",
            )
    else:
        pass

    from fastchat.serve.gradio_web_server import acknowledgment_md

    with gr.Accordion(
        "Citation",
        open=True,
    ):
        citation_md = """
            ### Citation
            Please cite the following paper if you find our leaderboard or dataset helpful.
            ```
            @misc{chiang2024chatbot,
                title={Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference},
                author={Wei-Lin Chiang and Lianmin Zheng and Ying Sheng and Anastasios Nikolas Angelopoulos and Tianle Li and Dacheng Li and Hao Zhang and Banghua Zhu and Michael Jordan and Joseph E. Gonzalez and Ion Stoica},
                year={2024},
                eprint={2403.04132},
                archivePrefix={arXiv},
                primaryClass={cs.AI}
            }
            """
        gr.Markdown(citation_md, elem_id="leaderboard_markdown")
        gr.Markdown(acknowledgment_md, elem_id="ack_markdown")

    return [md_1] + gr_plots


def build_demo(elo_results_file, leaderboard_table_file, arena_hard_leaderboard):
    from fastchat.serve.gradio_web_server import block_css

    text_size = gr.themes.sizes.text_lg
    # load theme from theme.json
    theme = gr.themes.Default.load("theme.json")
    # set text size to large
    theme.text_size = text_size
    theme.set(
        button_large_text_size="20px",
        button_small_text_size="20px",
        button_large_text_weight="100",
        button_small_text_weight="100",
        button_shadow="*shadow_drop_lg",
        button_shadow_hover="*shadow_drop_lg",
        checkbox_label_shadow="*shadow_drop_lg",
        button_shadow_active="*shadow_inset",
        button_secondary_background_fill="*primary_300",
        button_secondary_background_fill_dark="*primary_700",
        button_secondary_background_fill_hover="*primary_200",
        button_secondary_background_fill_hover_dark="*primary_500",
        button_secondary_text_color="*primary_800",
        button_secondary_text_color_dark="white",
    )

    with gr.Blocks(
        title="Chatbot Arena Leaderboard",
        # theme=gr.themes.Default(text_size=text_size),
        theme=theme,
        css=block_css,
    ) as demo:
        with gr.Tabs() as tabs:
            with gr.Tab("Leaderboard", id=0):
                leader_components = build_leaderboard_tab(
                    elo_results_file,
                    leaderboard_table_file,
                    arena_hard_leaderboard,
                    show_plot=True,
                    mirror=False,
                )

            with gr.Tab("Basic Stats", id=1):
                basic_components = build_basic_stats_tab()

        url_params = gr.JSON(visible=False)
        demo.load(
            load_demo,
            [url_params],
            basic_components + leader_components,
            js=get_window_url_params_js,
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--concurrency-count", type=int, default=10)
    parser.add_argument("--update-interval", type=int, default=300)
    parser.add_argument("--max-num-files", type=int)
    parser.add_argument("--elo-results-file", type=str)
    parser.add_argument("--leaderboard-table-file", type=str)
    parser.add_argument("--ban-ip-file", type=str)
    parser.add_argument("--exclude-model-names", type=str, nargs="+")
    parser.add_argument("--password", type=str, default=None, nargs="+")
    parser.add_argument("--arena-hard-leaderboard", type=str, default=None)
    args = parser.parse_args()

    logger = build_logger("monitor", "monitor.log")
    logger.info(f"args: {args}")

    if args.elo_results_file is None:  # Do live update
        update_thread = threading.Thread(
            target=update_worker,
            args=(
                args.max_num_files,
                args.update_interval,
                args.elo_results_file,
                args.ban_ip_file,
                args.exclude_model_names,
            ),
        )
        update_thread.start()

    demo = build_demo(
        args.elo_results_file, args.leaderboard_table_file, args.arena_hard_leaderboard
    )
    demo.queue(
        default_concurrency_limit=args.concurrency_count,
        status_update_rate=10,
        api_open=False,
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_threads=200,
        auth=(args.password[0], args.password[1]) if args.password else None,
    )
