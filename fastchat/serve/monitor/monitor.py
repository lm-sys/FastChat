# sudo apt install pkg-config libicu-dev
# pip install pytz gradio gdown plotly polyglot pyicu pycld2 tabulate

import argparse
import datetime
import pickle
from pytz import timezone
import os
import threading
import time

import gradio as gr
import pandas as pd

from fastchat.serve.monitor.basic_stats import report_basic_stats, get_log_files
from fastchat.serve.monitor.clean_battle_data import clean_battle_data
from fastchat.serve.monitor.elo_analysis import report_elo_analysis_results
from fastchat.utils import build_logger, get_window_url_params_js

logger = build_logger("monitor", "monitor.log")


basic_component_values = ["Loading ..."]
leader_component_values = [None, None, None, None, None]


table_css = """
table {
    line-height: 0em
}
"""


notebook_url = "https://colab.research.google.com/drive/1iI_IszGAwSMkdfUrIDI6NfTG7tGDDRxZ?usp=sharing"


def make_leaderboard_md(elo_results):
    leaderboard_md = f"""
# Leaderboard
[[Blog](https://lmsys.org/blog/2023-05-03-arena/)] [[GitHub]](https://github.com/lm-sys/FastChat) [[Twitter]](https://twitter.com/lmsysorg) [[Discord]](https://discord.gg/h6kCZb72G7)

We use the Elo rating system to calculate the relative performance of the models. You can view the voting data, basic analyses, and calculation procedure in this [notebook]({notebook_url}). We will periodically release new leaderboards. If you want to see more models, please help us [add them](https://github.com/lm-sys/FastChat/blob/main/docs/arena.md#how-to-add-a-new-model).
Last updated: {elo_results["last_updated_datetime"]}
{elo_results["leaderboard_table"]}
"""
    return leaderboard_md


def update_elo_components(max_num_files, elo_results_file):
    log_files = get_log_files(max_num_files)

    # Leaderboard
    if elo_results_file is None:
        battles = clean_battle_data(log_files)
        elo_results = report_elo_analysis_results(battles)

        leader_component_values[0] = make_leaderboard_md(elo_results)
        leader_component_values[1] = elo_results["win_fraction_heatmap"]
        leader_component_values[2] = elo_results["battle_count_heatmap"]
        leader_component_values[3] = elo_results["average_win_rate_bar"]
        leader_component_values[4] = elo_results["bootstrap_elo_rating"]

    # Basic stats
    basic_stats = report_basic_stats(log_files)
    basic_stats_md = ""
    basic_stats_md += "### Action Histogram\n"
    basic_stats_md += basic_stats["action_hist_md"] + "\n"
    basic_stats_md += "### Anony. Vote Histogram\n"
    basic_stats_md += basic_stats["anony_vote_hist_md"] + "\n"
    basic_stats_md += "### Model Call Histogram\n"
    basic_stats_md += basic_stats["model_hist_md"] + "\n"
    basic_stats_md += "### Model Call (Last 24 Hours)\n"
    basic_stats_md += basic_stats["num_chats_last_24_hours"] + "\n"
    date = datetime.datetime.now(tz=timezone("US/Pacific")).strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )
    basic_stats_md += f"\n\nLast updated: {date}"
    basic_component_values[0] = basic_stats_md


def update_worker(max_num_files, interval, elo_results_file):
    while True:
        update_elo_components(max_num_files, elo_results_file)
        time.sleep(interval)


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")
    return basic_component_values + leader_component_values


def build_basic_stats_tab():
    md = gr.Markdown()
    return [md]


def build_leaderboard_tab(elo_results_file):
    if elo_results_file is not None:
        with open(elo_results_file, "rb") as fin:
            elo_results = pickle.load(fin)

        md = make_leaderboard_md(elo_results)
        p1 = elo_results["win_fraction_heatmap"]
        p2 = elo_results["battle_count_heatmap"]
        p3 = elo_results["average_win_rate_bar"]
        p4 = elo_results["bootstrap_elo_rating"]
    else:
        md = "Loading ..."
        p1 = p2 = p3 = p4 = None

    leader_component_values[:] = [md, p1, p2, p3, p4]

    md_1 = gr.Markdown(md)
    gr.Markdown(
        f"""## More Statistics\n
Here, we have added some additional figures to show more statistics. The code for generating them is also included in this [notebook]({notebook_url}).
Please note that you may see different orders from different ranking methods. This is expected for models that perform similarly, as demonstrated by the confidence interval in the bootstrap figure. Going forward, we prefer the classical Elo calculation because of its scalability and interpretability. You can find more discussions in this blog [post](https://lmsys.org/blog/2023-05-03-arena/).
"""
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
                "#### Figure 3: Average Win Rate Against All Other Models (Assuming Uniform Sampling and No Ties)"
            )
            plot_3 = gr.Plot(p3, show_label=False)
        with gr.Column():
            gr.Markdown(
                "#### Figure 4: Bootstrap of Elo Estimates (1000 Rounds of Random Sampling)"
            )
            plot_4 = gr.Plot(p4, show_label=False)
    return [md_1, plot_1, plot_2, plot_3, plot_4]


def build_demo(elo_results_file):
    with gr.Blocks(
        title="Monitor",
        theme=gr.themes.Base(),
    ) as demo:
        with gr.Tabs() as tabs:
            with gr.Tab("Leaderboard", id=0):
                leader_components = build_leaderboard_tab(elo_results_file)

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
    args = parser.parse_args()
    logger.info(f"args: {args}")

    update_thread = threading.Thread(
        target=update_worker,
        args=(args.max_num_files, args.update_interval, args.elo_results_file),
    )
    update_thread.start()

    demo = build_demo(args.elo_results_file)
    demo.queue(
        concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False
    ).launch(
        server_name=args.host, server_port=args.port, share=args.share, max_threads=200
    )
