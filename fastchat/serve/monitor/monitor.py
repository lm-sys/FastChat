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

from fastchat.serve.monitor.basic_stats import report_basic_stats, get_log_files
from fastchat.serve.monitor.clean_battle_data import clean_battle_data
from fastchat.serve.monitor.elo_analysis import report_elo_analysis_results
from fastchat.utils import build_logger, get_window_url_params_js


notebook_url = (
    "https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH"
)

basic_component_values = [None] * 6
leader_component_values = [None] * 5


def make_default_md_1(arena_df, elo_results, mirror=False):
    link_color = "#1976D2"  # This color should be clear in both light and dark mode
    leaderboard_md = f"""
    # 🏆 LMSYS Chatbot Arena Leaderboard 
    <a href='https://lmsys.org/blog/2023-05-03-arena/' style='color: {link_color}; text-decoration: none;'>Blog</a> |
    <a href='https://arxiv.org/abs/2403.04132' style='color: {link_color}; text-decoration: none;'>Paper</a> |
    <a href='https://github.com/lm-sys/FastChat' style='color: {link_color}; text-decoration: none;'>GitHub</a> |
    <a href='https://github.com/lm-sys/FastChat/blob/main/docs/dataset_release.md' style='color: {link_color}; text-decoration: none;'>Dataset</a> |
    <a href='https://twitter.com/lmsysorg' style='color: {link_color}; text-decoration: none;'>Twitter</a> |
    <a href='https://discord.gg/HSWAKCrnFx' style='color: {link_color}; text-decoration: none;'>Discord</a>
    """

    return leaderboard_md


def make_default_md_2(arena_df, elo_results, mirror=False):
    mirror_str = "<span style='color: red; font-weight: bold'>This is a mirror of the live leaderboard created and maintained by the <a href='https://lmsys.org' style='color: red; text-decoration: none;'>LMSYS Organization</a>. Please link to <a href='https://leaderboard.lmsys.org' style='color: #B00020; text-decoration: none;'>leaderboard.lmsys.org</a> for citation purposes.</span>"
    leaderboard_md = f"""
    {mirror_str if mirror else ""}
    
    LMSYS Chatbot Arena is a crowdsourced open platform for LLM evals. We've collected over 800,000 human pairwise comparisons to rank LLMs with the Bradley-Terry model and display the model ratings in Elo-scale.
    You can find more details in our paper. **Chatbot arena is dependent on community participation, please contribute by casting your vote!**
    """

    return leaderboard_md


def make_arena_leaderboard_md(arena_df, last_updated_time):
    total_votes = sum(arena_df["num_battles"]) // 2
    total_models = len(arena_df)
    space = "&nbsp;&nbsp;&nbsp;"

    leaderboard_md = f"""
Total #models: **{total_models}**.{space} Total #votes: **{"{:,}".format(total_votes)}**.{space} Last updated: {last_updated_time}.

📣 **NEW!** View leaderboard for different categories (e.g., coding, long user query)! This is still in preview and subject to change.

Code to recreate leaderboard tables and plots in this [notebook]({notebook_url}). You can contribute your vote at [chat.lmsys.org](https://chat.lmsys.org)!

***Rank (UB)**: model's ranking (upper-bound), defined by one + the number of models that are statistically better than the target model.
Model A is statistically better than model B when A's lower-bound score is greater than B's upper-bound score (in 95% confidence interval).
See Figure 1 below for visualization of the confidence intervals of model scores.
"""
    return leaderboard_md


def make_category_arena_leaderboard_md(arena_df, arena_subset_df, name="Overall"):
    total_votes = sum(arena_df["num_battles"]) // 2
    total_models = len(arena_df)
    space = "&nbsp;&nbsp;&nbsp;"
    total_subset_votes = sum(arena_subset_df["num_battles"]) // 2
    total_subset_models = len(arena_subset_df)
    leaderboard_md = f"""### {cat_name_to_explanation[name]}
#### {space} #models: **{total_subset_models} ({round(total_subset_models/total_models *100)}%)** {space} #votes: **{"{:,}".format(total_subset_votes)} ({round(total_subset_votes/total_votes * 100)}%)**{space}
"""
    return leaderboard_md


def make_full_leaderboard_md(elo_results):
    leaderboard_md = """
Three benchmarks are displayed: **Arena Elo**, **MT-Bench** and **MMLU**.
- [Chatbot Arena](https://chat.lmsys.org/?arena) - a crowdsourced, randomized battle platform. We use 500K+ user votes to compute model strength.
- [MT-Bench](https://arxiv.org/abs/2306.05685): a set of challenging multi-turn questions. We use GPT-4 to grade the model responses.
- [MMLU](https://arxiv.org/abs/2009.03300) (5-shot): a test to measure a model's multitask accuracy on 57 tasks.

💻 Code: The MT-bench scores (single-answer grading on a scale of 10) are computed by [fastchat.llm_judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge).
The MMLU scores are mostly computed by [InstructEval](https://github.com/declare-lab/instruct-eval).
Higher values are better for all benchmarks. Empty cells mean not available.
"""
    return leaderboard_md


def make_leaderboard_md_live(elo_results):
    leaderboard_md = f"""
# Leaderboard
Last updated: {elo_results["last_updated_datetime"]}
{elo_results["leaderboard_table"]}
"""
    return leaderboard_md


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


def get_full_table(arena_df, model_table_df):
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
        row.append(model_table_df.iloc[i]["MT-bench (score)"])
        row.append(model_table_df.iloc[i]["MMLU"])
        # Organization
        row.append(model_table_df.iloc[i]["Organization"])
        # license
        row.append(model_table_df.iloc[i]["License"])

        values.append(row)
    values.sort(key=lambda x: -x[1] if not np.isnan(x[1]) else 1e9)
    return values


def create_ranking_str(ranking, ranking_difference):
    if ranking_difference > 0:
        return f"{int(ranking)} \u2191"
    elif ranking_difference < 0:
        return f"{int(ranking)} \u2193"
    else:
        return f"{int(ranking)}"


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


def highlight_top_models(df):
    def highlight_max_rank(s):
        # Pastel Yellow with transparency, rgba(red, green, blue, alpha)
        highlight_color = "rgba(255, 255, 128, 0.2)"  # 50% transparent
        if int(s["Rank* (UB)"].replace("↑", "").replace("↓", "")) == 1:
            return [f"background-color: {highlight_color}" for _ in s]
        else:
            return ["" for _ in s]

    # Apply and return the styled DataFrame
    return df.apply(highlight_max_rank, axis=1)


def get_arena_table(arena_df, model_table_df, arena_subset_df=None):
    arena_df = arena_df.sort_values(
        by=["final_ranking", "rating"], ascending=[True, False]
    )
    arena_df["final_ranking"] = recompute_final_ranking(arena_df)
    arena_df = arena_df.sort_values(by=["final_ranking"], ascending=True)

    # sort by rating
    if arena_subset_df is not None:
        # filter out models not in the arena_df
        arena_subset_df = arena_subset_df[arena_subset_df.index.isin(arena_df.index)]
        arena_subset_df = arena_subset_df.sort_values(by=["rating"], ascending=False)
        arena_subset_df["final_ranking"] = recompute_final_ranking(arena_subset_df)
        # keep only the models in the subset in arena_df and recompute final_ranking
        arena_df = arena_df[arena_df.index.isin(arena_subset_df.index)]
        # recompute final ranking
        arena_df["final_ranking"] = recompute_final_ranking(arena_df)

        # assign ranking by the order
        arena_subset_df["final_ranking_no_tie"] = range(1, len(arena_subset_df) + 1)
        arena_df["final_ranking_no_tie"] = range(1, len(arena_df) + 1)
        # join arena_df and arena_subset_df on index
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

    values = []
    for i in range(len(arena_df)):
        row = []
        model_key = arena_df.index[i]
        try:  # this is a janky fix for where the model key is not in the model table (model table and arena table dont contain all the same models)
            model_name = model_table_df[model_table_df["key"] == model_key][
                "Model"
            ].values[0]
            # rank
            ranking = arena_df.iloc[i].get("final_ranking") or i + 1
            row.append(ranking)
            if arena_subset_df is not None:
                row.append(arena_df.iloc[i].get("ranking_difference") or 0)
            # model display name
            row.append(model_name)
            # elo rating
            row.append(round(arena_df.iloc[i]["rating"]))
            upper_diff = round(
                arena_df.iloc[i]["rating_q975"] - arena_df.iloc[i]["rating"]
            )
            lower_diff = round(
                arena_df.iloc[i]["rating"] - arena_df.iloc[i]["rating_q025"]
            )
            row.append(f"+{upper_diff}/-{lower_diff}")
            # num battles
            row.append(round(arena_df.iloc[i]["num_battles"]))
            # Organization
            row.append(
                model_table_df[model_table_df["key"] == model_key][
                    "Organization"
                ].values[0]
            )
            # license
            row.append(
                model_table_df[model_table_df["key"] == model_key]["License"].values[0]
            )
            cutoff_date = model_table_df[model_table_df["key"] == model_key][
                "Knowledge cutoff date"
            ].values[0]
            if cutoff_date == "-":
                row.append("Unknown")
            else:
                row.append(cutoff_date)
            values.append(row)
        except Exception as e:
            print(f"{model_key} - {e}")
    return values


key_to_category_name = {
    "full": "Overall",
    "coding": "Coding",
    "long_user": "Longer Query",
    "english": "English",
    "chinese": "Chinese",
    "french": "French",
    "no_tie": "Exclude Ties",
    "no_short": "Exclude Short Query (< 5 tokens)",
    "no_refusal": "Exclude Refusal",
}
cat_name_to_explanation = {
    "Overall": "Overall Questions",
    "Coding": "Coding: whether conversation contains code snippets",
    "Longer Query": "Longer Query (>= 500 tokens)",
    "English": "English Prompts",
    "Chinese": "Chinese Prompts",
    "French": "French Prompts",
    "Exclude Ties": "Exclude Ties and Bothbad",
    "Exclude Short Query (< 5 tokens)": "Exclude Short User Query (< 5 tokens)",
    "Exclude Refusal": 'Exclude model responses with refusal (e.g., "I cannot answer")',
}


def build_leaderboard_tab(
    elo_results_file, leaderboard_table_file, show_plot=False, mirror=False
):
    arena_dfs = {}
    category_elo_results = {}
    if elo_results_file is None:  # Do live update
        default_md = "Loading ..."
        p1 = p2 = p3 = p4 = None
    else:
        with open(elo_results_file, "rb") as fin:
            elo_results = pickle.load(fin)
            last_updated_time = None
            if "full" in elo_results:
                last_updated_time = elo_results["full"]["last_updated_datetime"].split(
                    " "
                )[0]
                for k in key_to_category_name.keys():
                    if k not in elo_results:
                        continue
                    arena_dfs[key_to_category_name[k]] = elo_results[k][
                        "leaderboard_table_df"
                    ]
                    category_elo_results[key_to_category_name[k]] = elo_results[k]

        p1 = category_elo_results["Overall"]["win_fraction_heatmap"]
        p2 = category_elo_results["Overall"]["battle_count_heatmap"]
        p3 = category_elo_results["Overall"]["bootstrap_elo_rating"]
        p4 = category_elo_results["Overall"]["average_win_rate_bar"]
        arena_df = arena_dfs["Overall"]
        default_md = make_default_md_1(
            arena_df, category_elo_results["Overall"], mirror=mirror
        )
        default_md_2 = make_default_md_2(
            arena_df, category_elo_results["Overall"], mirror=mirror
        )

    with gr.Row():
        with gr.Column(scale=4):
            md_1 = gr.Markdown(default_md, elem_id="leaderboard_markdown")
        with gr.Column(scale=1):
            vote_button = gr.Button("Vote!", link="https://chat.lmsys.org")
    md2 = gr.Markdown(default_md_2, elem_id="leaderboard_markdown")
    if leaderboard_table_file:
        data = load_leaderboard_table_csv(leaderboard_table_file)
        model_table_df = pd.DataFrame(data)

        with gr.Tabs() as tabs:
            # arena table
            arena_table_vals = get_arena_table(arena_df, model_table_df)
            with gr.Tab("Arena", id=0):
                md = make_arena_leaderboard_md(arena_df, last_updated_time)
                gr.Markdown(md, elem_id="leaderboard_markdown")
                with gr.Row():
                    with gr.Column(scale=2):
                        category_dropdown = gr.Dropdown(
                            choices=list(arena_dfs.keys()),
                            label="Category",
                            value="Overall",
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
                        "Arena Elo",
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
                        "str",
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
                    height=700,
                    column_widths=[70, 190, 100, 100, 90, 130, 150, 100],
                    wrap=True,
                )

                gr.Markdown(
                    f"""Note: in each category, we exclude models with fewer than 500 votes as their confidence intervals can be large.""",
                    elem_id="leaderboard_markdown",
                )

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
                            plot_1 = gr.Plot(
                                p1, show_label=False, elem_id="plot-container"
                            )
                        with gr.Column():
                            gr.Markdown(
                                "#### Figure 4: Battle Count for Each Combination of Models (without Ties)",
                                elem_id="plot-title",
                            )
                            plot_2 = gr.Plot(p2, show_label=False)
            with gr.Tab("Full Leaderboard", id=1):
                md = make_full_leaderboard_md(elo_results)
                gr.Markdown(md, elem_id="leaderboard_markdown")
                full_table_vals = get_full_table(arena_df, model_table_df)
                gr.Dataframe(
                    headers=[
                        "Model",
                        "Arena Elo",
                        "MT-bench",
                        "MMLU",
                        "Organization",
                        "License",
                    ],
                    datatype=["markdown", "number", "number", "number", "str", "str"],
                    value=full_table_vals,
                    elem_id="full_leaderboard_dataframe",
                    column_widths=[200, 100, 100, 100, 150, 150],
                    height=700,
                    wrap=True,
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

    def update_leaderboard_df(arena_table_vals):
        elo_datarame = pd.DataFrame(
            arena_table_vals,
            columns=[
                "Rank* (UB)",
                "Delta",
                "Model",
                "Arena Elo",
                "95% CI",
                "Votes",
                "Organization",
                "License",
                "Knowledge Cutoff",
            ],
        )

        # goal: color the rows based on the rank with styler
        def highlight_max(s):
            # all items in S which contain up arrow should be green, down arrow should be red, otherwise black
            return [
                "color: green; font-weight: bold"
                if "\u2191" in v
                else "color: red; font-weight: bold"
                if "\u2193" in v
                else ""
                for v in s
            ]

        def highlight_rank_max(s):
            return [
                "color: green; font-weight: bold"
                if v > 0
                else "color: red; font-weight: bold"
                if v < 0
                else ""
                for v in s
            ]

        return elo_datarame.style.apply(highlight_max, subset=["Rank* (UB)"]).apply(
            highlight_rank_max, subset=["Delta"]
        )

    def update_leaderboard_and_plots(category):
        arena_subset_df = arena_dfs[category]
        arena_subset_df = arena_subset_df[arena_subset_df["num_battles"] > 500]
        elo_subset_results = category_elo_results[category]
        arena_df = arena_dfs["Overall"]
        arena_values = get_arena_table(
            arena_df,
            model_table_df,
            arena_subset_df=arena_subset_df if category != "Overall" else None,
        )
        if category != "Overall":
            arena_values = update_leaderboard_df(arena_values)
            # arena_values = highlight_top_models(arena_values)
            arena_values = gr.Dataframe(
                headers=[
                    "Rank* (UB)",
                    "Delta",
                    "Model",
                    "Arena Elo",
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
                height=700,
                column_widths=[70, 70, 200, 90, 100, 90, 120, 150, 100],
                wrap=True,
            )
        else:
            # not_arena_values = pd.DataFrame(arena_values, columns=["Rank* (UB)",
            #         "Model",
            #         "Arena Elo",
            #         "95% CI",
            #         "Votes",
            #         "Organization",
            #         "License",
            #         "Knowledge Cutoff",],
            #         )
            # arena_values = highlight_top_models(not_arena_values.style)
            arena_values = gr.Dataframe(
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
                    "str",
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
                height=700,
                column_widths=[70, 190, 100, 100, 90, 140, 150, 100],
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

    category_dropdown.change(
        update_leaderboard_and_plots,
        inputs=[category_dropdown],
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

    if show_plot:
        return [md_1, plot_1, plot_2, plot_3, plot_4]
    return [md_1]


def build_demo(elo_results_file, leaderboard_table_file):
    from fastchat.serve.gradio_web_server import block_css

    text_size = gr.themes.sizes.text_lg
    # load theme from theme.json
    theme = gr.themes.Default.load("theme.json")
    # set text size to large
    theme.text_size = text_size
    theme.set(
        button_large_text_size="40px",
        button_small_text_size="40px",
        button_large_text_weight="1000",
        button_small_text_weight="1000",
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

    demo = build_demo(args.elo_results_file, args.leaderboard_table_file)
    demo.queue(
        default_concurrency_limit=args.concurrency_count,
        status_update_rate=10,
        api_open=False,
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_threads=200,
    )
