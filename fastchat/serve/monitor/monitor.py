import argparse
import datetime
from pytz import timezone
import os
import threading
import time

import gradio as gr

from fastchat.serve.monitor.basic_stats import report_basic_stats
from fastchat.utils import build_logger, get_window_url_params_js

logger = build_logger("monitor", "monitor.log")


basic_stats_md = ""
leaderboard_md = ""


def get_log_files():
    dates = []
    for month in [4]:
        for day in range(24, 32):
            dates.append(f"2023-{month:02d}-{day:02d}")
    num_servers = 10

    filenames = []
    for d in dates:
        for i in range(num_servers):
            name = f"/home/ubuntu/fastchat_logs/server{i}/{d}-conv.json"
            if os.path.exists(name):
                filenames.append(name)

    return filenames


def update_md_content():
    global basic_stats_md, leaderboard_md
    log_files = get_log_files()

    # Basic stats
    basic_stats_md = "Updating..."
    basic_stats = report_basic_stats(log_files)
    basic_stats_tmp = ""
    basic_stats_tmp += "### Action Histogram\n"
    basic_stats_tmp += basic_stats["action_hist_md"] + "\n"
    basic_stats_tmp += "### Anony. Vote Histogram\n"
    basic_stats_tmp += basic_stats["anony_vote_hist_md"] + "\n"
    basic_stats_tmp += "### Model Call Histogram\n"
    basic_stats_tmp += basic_stats["model_hist_md"] + "\n"
    basic_stats_tmp += "### Model Call (Last 24 Hours)\n"
    basic_stats_tmp += basic_stats["num_chats_last_24_hours"] + "\n"
    date = datetime.datetime.now(tz=timezone('US/Pacific')).strftime("%Y-%m-%d %H:%M:%S %Z")
    basic_stats_tmp += f"\n\nLast update: {date}"
    basic_stats_md = basic_stats_tmp


def update_worker(interval):
    while True:
        update_md_content()
        time.sleep(interval)


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")
    return basic_stats_md, leaderboard_md


def build_basic_stats_tab():
    md = gr.Markdown()
    return md


def build_leaderboard_tab():
    md = gr.Markdown()
    return md


def build_demo():
    with gr.Blocks(
        title="Monitor",
        theme=gr.themes.Base(),
    ) as demo:
        with gr.Tabs() as tabs:
            with gr.Tab("Basic Stats", id=0):
                basic_gr_md = build_basic_stats_tab()

            with gr.Tab("Leaderboard", id=1):
                leader_gr_md = build_leaderboard_tab()

        url_params = gr.JSON(visible=False)
        demo.load(
            load_demo,
            [url_params],
            [basic_gr_md, leader_gr_md],
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
    args = parser.parse_args()
    logger.info(f"args: {args}")

    update_thread = threading.Thread(target=update_worker, args=(args.update_interval,))
    update_thread.start()

    demo = build_demo()
    demo.queue(
        concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False
    ).launch(
        server_name=args.host, server_port=args.port, share=args.share, max_threads=200
    )
