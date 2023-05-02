import argparse
import datetime
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
leader_component_values = [None, None, None, None]


table_css = """
table {
    line-height: 0em
}
"""

model_info = {
    "vicuna-13b": ("https://vicuna.lmsys.org", "a chat assistant fine-tuned from LLaMA on user-shared conversations by LMSYS"),
    "koala-13b": ("https://bair.berkeley.edu/blog/2023/04/03/koala", "a dialogue model for academic research by BAIR"),
    "fastchat-t5-3b": ("https://huggingface.co/lmsys/fastchat-t5-3b-v1.0", "a chat assistant fine-tuned from FLAN-T5 by LMSYS"),
    "oasst-pythia-12b": ("https://open-assistant.io", "an Open Assistant for everyone by LAION"),
    "chatglm-6b": ("https://chatglm.cn/blog", "an open bilingual dialogue language model by Tsinghua University"),
    "stablelm-tuned-alpha-7b": ("https://github.com/stability-AI/stableLM", "Stability AI language models"),
    "alpaca-13b": ("https://crfm.stanford.edu/2023/03/13/alpaca.html", "a model fine-tuned from LLaMA on instruction-following demonstrations by Stanford"),
    "llama-13b": ("https://arxiv.org/abs/2302.13971", "open and efficient foundation language models by Meta"),
    "dolly-v2-12b": ("https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm", "an instruction-tuned open large language model by Databricks"),
}


def gen_leaderboard_md(rating):
    models = list(rating.keys())
    models.sort(key=lambda k: -rating[k])

    emoji_dict = {
        1: "🥇",
        2: "🥈",
        3: "🥉",
    }

    md = "| Rank | Model | Elo Rating | Description |\n"
    md += "| --- | --- | --- | --- |\n"
    for i, model in enumerate(models):
        rank = i + 1
        link, desc = model_info[model]
        emoji = emoji_dict.get(rank, "")
        md += f"| {rank} | {emoji} [{model}]({link}) | {rating[model]:.0f} | {desc} |\n"

    return md


def update_md_content(max_num_files):
    log_files = get_log_files(max_num_files)

    # Leaderboard
    results = report_elo_analysis_results(log_files)
    leader_component_values[0] = ("""
# Leaderboard
We use the Elo rating system to calculate the relative performance of the models.
This [notebook](https://colab.research.google.com/drive/1lAQ9cKVErXI1rEYq7hTKNaCQ5Q8TzrI5?usp=sharing) shares the voting data, basic analyses, and computation procedure. (Update date: May 1, 2023)
""" + gen_leaderboard_md(results["elo_rating"]))

    leader_component_values[1] = results["win_fraction_heatmap"]
    leader_component_values[2] = results["battle_count_heatmap"]
    leader_component_values[3] = results["average_win_rate_bar"]

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
    date = datetime.datetime.now(tz=timezone('US/Pacific')).strftime("%Y-%m-%d %H:%M:%S %Z")
    basic_stats_md += f"\n\nLast update: {date}"
    basic_component_values[0] = basic_stats_md


def update_worker(max_num_files, interval):
    while True:
        update_md_content(max_num_files)
        time.sleep(interval)


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")
    return basic_component_values + leader_component_values


def build_basic_stats_tab():
    md = gr.Markdown()
    return [md]


def build_leaderboard_tab():
    md_1 = gr.Markdown()
    gr.Markdown("### More Statistics")
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Figure 1: Fraction of model A wins for all non-tied A vs. B battles")
            plot_1 = gr.Plot()
        with gr.Column():
            gr.Markdown("#### Figure 2: Battle Count for Each Combination of Models (without Ties)")
            plot_2 = gr.Plot()
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Figure 3: Average Win Rate Against All Other Models (Assuming Uniform Sampling and No Ties")
            plot_3 = gr.Plot()
    return [md_1, plot_1, plot_2, plot_3]


def build_demo():
    with gr.Blocks(
        title="Monitor",
        css=table_css,
        theme=gr.themes.Base(),
    ) as demo:
        with gr.Tabs() as tabs:
            with gr.Tab("Leaderboard", id=0):
                leader_components = build_leaderboard_tab()

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
    parser.add_argument("--update-interval", type=int, default=1800)
    parser.add_argument("--max-num-files", type=int)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    update_thread = threading.Thread(target=update_worker,
        args=(args.max_num_files, args.update_interval))
    update_thread.start()

    demo = build_demo()
    demo.queue(
        concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False
    ).launch(
        server_name=args.host, server_port=args.port, share=args.share, max_threads=200
    )
