import argparse
import datetime
from pytz import timezone
import threading
import time

import gradio as gr

from fastchat.serve.monitor.elo_rating_linear_update import print_ratings_linear_update
from fastchat.serve.monitor.elo_rating_mle import print_ratings_mle
from fastchat.utils import build_logger

logger = build_logger("monitor", "monitor.log")


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log("url_params", url_params);
    return url_params;
    }
"""

md_content = ""


def get_log_files():
    dates = ["2023-04-24", "2023-04-25", "2023-04-26", "2023-04-27"]
    num_servers = 10

    filenames = []
    for i in range(num_servers):
        for d in dates:
            filenames.append(f"/home/Ying/fastchat_logs/server{i}/{d}-conv.json")

    return filenames


def update_md_content():
    global md_content
    log_files = get_log_files()

    md_content = print_ratings_linear_update(log_files)
    # md_content = print_ratings_mle(log_files)
    md_content = md_content.replace("---:", "---")

    date = datetime.datetime.now(tz=timezone('US/Pacific'))
    date = date.strftime("%Y-%m-%d %H:%M:%S %Z")
    md_content += f"\nLast update: {date}"


def update_worker(interval):
    while True:
        time.sleep(interval)
        update_md_content()


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")
    return md_content


def build_demo():
    with gr.Blocks(
        title="Monitor",
        theme=gr.themes.Base(),
    ) as demo:
        url_params = gr.JSON(visible=False)

        leaderboard = gr.Markdown()

        demo.load(
            load_demo,
            [url_params],
            [leaderboard],
            _js=get_window_url_params,
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--concurrency-count", type=int, default=10)
    parser.add_argument("--update-interval", type=int, default=3600)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    update_md_content()
    update_thread = threading.Thread(target=update_worker, args=(args.update_interval,))
    update_thread.start()

    demo = build_demo()
    demo.queue(
        concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False
    ).launch(
        server_name=args.host, server_port=args.port, share=args.share, max_threads=200
    )
