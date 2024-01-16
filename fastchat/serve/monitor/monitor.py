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


def make_default_md(arena_df, elo_results):
    total_votes = sum(arena_df["num_battles"]) // 2
    total_models = len(arena_df)

    leaderboard_md = f"""
# 🏆 LMSYS Chatbot Arena Leaderboard
| [Vote](https://chat.lmsys.org) | [Blog](https://lmsys.org/blog/2023-05-03-arena/) | [GitHub](https://github.com/lm-sys/FastChat) | [Paper](https://arxiv.org/abs/2306.05685) | [Dataset](https://github.com/lm-sys/FastChat/blob/main/docs/dataset_release.md) | [Twitter](https://twitter.com/lmsysorg) | [Discord](https://discord.gg/HSWAKCrnFx) |

LMSYS [Chatbot Arena](https://lmsys.org/blog/2023-05-03-arena/) is a crowdsourced open platform for LLM evals.
We've collected over **200,000** human preference votes to rank LLMs with the Elo ranking system.
"""
    return leaderboard_md


def make_arena_leaderboard_md(arena_df):
    total_votes = sum(arena_df["num_battles"]) // 2
    total_models = len(arena_df)

    leaderboard_md = f"""
Total #models: **{total_models}**. Total #votes: **{total_votes}**. Last updated: Jan 9, 2024.

Contribute your vote 🗳️ at [chat.lmsys.org](https://chat.lmsys.org)! Find more analysis in the [notebook]({notebook_url}).
"""
    return leaderboard_md


def make_full_leaderboard_md(elo_results):
    leaderboard_md = f"""
Two more benchmarks are displayed: **MT-Bench** and **MMLU**.
- [MT-Bench](https://arxiv.org/abs/2306.05685): a set of challenging multi-turn questions. We use GPT-4 to grade the model responses.
- [MMLU](https://arxiv.org/abs/2009.03300) (5-shot): a test to measure a model's multitask accuracy on 57 tasks.
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
        elo_results = report_elo_analysis_results(battles)

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


def get_arena_table(arena_df, model_table_df):
    # sort by rating
    arena_df = arena_df.sort_values(by=["rating"], ascending=False)
    values = []
    for i in range(len(arena_df)):
        row = []
        model_key = arena_df.index[i]
        model_name = model_table_df[model_table_df["key"] == model_key]["Model"].values[
            0
        ]

        # rank
        row.append(i + 1)
        # model display name
        row.append(model_name)
        # elo rating
        row.append(round(arena_df.iloc[i]["rating"]))
        upper_diff = round(arena_df.iloc[i]["rating_q975"] - arena_df.iloc[i]["rating"])
        lower_diff = round(arena_df.iloc[i]["rating"] - arena_df.iloc[i]["rating_q025"])
        row.append(f"+{upper_diff}/-{lower_diff}")
        # num battles
        row.append(round(arena_df.iloc[i]["num_battles"]))
        # Organization
        row.append(
            model_table_df[model_table_df["key"] == model_key]["Organization"].values[0]
        )
        # license
        row.append(
            model_table_df[model_table_df["key"] == model_key]["License"].values[0]
        )

        values.append(row)
    return values


def build_leaderboard_tab(elo_results_file, leaderboard_table_file, show_plot=False):
    if elo_results_file is None:  # Do live update
        default_md = "Loading ..."
        p1 = p2 = p3 = p4 = None
    else:
        with open(elo_results_file, "rb") as fin:
            elo_results = pickle.load(fin)

        p1 = elo_results["win_fraction_heatmap"]
        p2 = elo_results["battle_count_heatmap"]
        p3 = elo_results["bootstrap_elo_rating"]
        p4 = elo_results["average_win_rate_bar"]
        arena_df = elo_results["leaderboard_table_df"]
        default_md = make_default_md(arena_df, elo_results)

    md_1 = gr.Markdown(default_md, elem_id="leaderboard_markdown")
    if leaderboard_table_file:
        data = load_leaderboard_table_csv(leaderboard_table_file)
        model_table_df = pd.DataFrame(data)

        with gr.Tabs() as tabs:
            # arena table
            arena_table_vals = get_arena_table(arena_df, model_table_df)
            with gr.Tab("Arena Elo", id=0):
                md = make_arena_leaderboard_md(arena_df)
                gr.Markdown(md, elem_id="leaderboard_markdown")
                gr.Dataframe(
                    headers=[
                        "Rank",
                        "🤖 Model",
                        "⭐ Arena Elo",
                        "📊 95% CI",
                        "🗳️ Votes",
                        "Organization",
                        "License",
                    ],
                    datatype=[
                        "str",
                        "markdown",
                        "number",
                        "str",
                        "number",
                        "str",
                        "str",
                    ],
                    value=arena_table_vals,
                    elem_id="arena_leaderboard_dataframe",
                    height=700,
                    column_widths=[50, 200, 100, 100, 100, 150, 150],
                    wrap=True,
                )
            with gr.Tab("Full Leaderboard", id=1):
                md = make_full_leaderboard_md(elo_results)
                gr.Markdown(md, elem_id="leaderboard_markdown")
                full_table_vals = get_full_table(arena_df, model_table_df)
                gr.Dataframe(
                    headers=[
                        "🤖 Model",
                        "⭐ Arena Elo",
                        "📈 MT-bench",
                        "📚 MMLU",
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

    leader_component_values[:] = [default_md, p1, p2, p3, p4]

    if show_plot:
        gr.Markdown(
            f"""## More Statistics for Chatbot Arena\n
Below are figures for more statistics. The code for generating them is also included in this [notebook]({notebook_url}).
You can find more discussions in this blog [post](https://lmsys.org/blog/2023-12-07-leaderboard/).
    """,
            elem_id="leaderboard_markdown",
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    "#### Figure 1: Fraction of Model A Wins for All Non-tied A vs. B Battles"
                )
                plot_1 = gr.Plot(p1, show_label=False)
            with gr.Column():
                gr.Markdown(
                    "#### Figure 2: Battle Count for Each Combination of Models (without Ties)"
                )
                plot_2 = gr.Plot(p2, show_label=False)
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    "#### Figure 3: Bootstrap of Elo Estimates (1000 Rounds of Random Sampling)"
                )
                plot_3 = gr.Plot(p3, show_label=False)
            with gr.Column():
                gr.Markdown(
                    "#### Figure 4: Average Win Rate Against All Other Models (Assuming Uniform Sampling and No Ties)"
                )
                plot_4 = gr.Plot(p4, show_label=False)

    from fastchat.serve.gradio_web_server import acknowledgment_md

    gr.Markdown(acknowledgment_md)

    if show_plot:
        return [md_1, plot_1, plot_2, plot_3, plot_4]
    return [md_1]


def build_demo(elo_results_file, leaderboard_table_file):
    from fastchat.serve.gradio_web_server import block_css

    text_size = gr.themes.sizes.text_lg

    with gr.Blocks(
        title="Monitor",
        theme=gr.themes.Base(text_size=text_size),
        css=block_css,
    ) as demo:
        with gr.Tabs() as tabs:
            with gr.Tab("Leaderboard", id=0):
                leader_components = build_leaderboard_tab(
                    elo_results_file,
                    leaderboard_table_file,
                    show_plot=True,
                )

            with gr.Tab("Basic Stats", id=1):
                basic_components = build_basic_stats_tab()

        url_params = gr.JSON(visible=False)
        demo.load(
            load_demo,
            [url_params],
            basic_components + leader_components,
            _js=get_window_url_params_js,
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
        concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False
    ).launch(
        server_name=args.host, server_port=args.port, share=args.share, max_threads=200
    )
