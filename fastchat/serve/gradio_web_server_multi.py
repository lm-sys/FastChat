"""
The gradio demo server with multiple tabs.
It supports chatting with a single model or chatting with two models side-by-side.
"""

import argparse
import pickle

import gradio as gr

from fastchat.serve.gradio_block_arena_anony import (
    build_side_by_side_ui_anony,
    load_demo_side_by_side_anony,
    set_global_vars_anony,
)
from fastchat.serve.gradio_block_arena_named import (
    build_side_by_side_ui_named,
    load_demo_side_by_side_named,
    set_global_vars_named,
)
from fastchat.serve.gradio_patch import Chatbot as grChatbot
from fastchat.serve.gradio_web_server import (
    set_global_vars,
    block_css,
    build_single_model_ui,
    get_model_list,
    load_demo_single,
)
from fastchat.serve.monitor.monitor import build_leaderboard_tab
from fastchat.utils import build_logger, get_window_url_params_js


logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")
    selected = 0
    if "arena" in url_params:
        selected = 1
    elif "compare" in url_params:
        selected = 2
    elif "leaderboard" in url_params:
        selected = 3
    single_updates = load_demo_single(models, url_params)

    models_anony = models
    # Only enable these models in anony battles.
    if args.add_chatgpt:
        models_anony = ["gpt-4", "gpt-3.5-turbo"] + models_anony
    if args.add_claude:
        models_anony = ["claude-v1"] + models_anony
    if args.add_bard:
        models_anony = ["bard"] + models_anony

    side_by_side_anony_updates = load_demo_side_by_side_anony(models_anony, url_params)
    side_by_side_named_updates = load_demo_side_by_side_named(models, url_params)
    return (
        (gr.Tabs.update(selected=selected),)
        + single_updates
        + side_by_side_anony_updates
        + side_by_side_named_updates
    )


def build_demo(models, elo_results_file):
    with gr.Blocks(
        title="Chat with Open Large Language Models",
        theme=gr.themes.Base(),
        css=block_css,
    ) as demo:
        with gr.Tabs() as tabs:
            with gr.Tab("Single Model", id=0):
                (
                    a_state,
                    a_model_selector,
                    a_chatbot,
                    a_textbox,
                    a_send_btn,
                    a_button_row,
                    a_parameter_row,
                ) = build_single_model_ui(models)
                a_list = [
                    a_state,
                    a_model_selector,
                    a_chatbot,
                    a_textbox,
                    a_send_btn,
                    a_button_row,
                    a_parameter_row,
                ]

            with gr.Tab("Chatbot Arena (battle)", id=1):
                (
                    b_states,
                    b_model_selectors,
                    b_chatbots,
                    b_textbox,
                    b_send_btn,
                    b_button_row,
                    b_button_row2,
                    b_parameter_row,
                ) = build_side_by_side_ui_anony(models)
                b_list = (
                    b_states
                    + b_model_selectors
                    + b_chatbots
                    + [
                        b_textbox,
                        b_send_btn,
                        b_button_row,
                        b_button_row2,
                        b_parameter_row,
                    ]
                )

            with gr.Tab("Chatbot Arena (side-by-side)", id=2):
                (
                    c_states,
                    c_model_selectors,
                    c_chatbots,
                    c_textbox,
                    c_send_btn,
                    c_button_row,
                    c_button_row2,
                    c_parameter_row,
                ) = build_side_by_side_ui_named(models)
                c_list = (
                    c_states
                    + c_model_selectors
                    + c_chatbots
                    + [
                        c_textbox,
                        c_send_btn,
                        c_button_row,
                        c_button_row2,
                        c_parameter_row,
                    ]
                )

            if elo_results_file:
                with gr.Tab("Leaderboard", id=3):
                    build_leaderboard_tab(elo_results_file)

        url_params = gr.JSON(visible=False)

        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [tabs] + a_list + b_list + c_list,
                _js=get_window_url_params_js,
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=10)
    parser.add_argument(
        "--model-list-mode", type=str, default="once", choices=["once"],
    )
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--moderate", action="store_true", help="Enable content moderation"
    )
    parser.add_argument(
        "--add-chatgpt",
        action="store_true",
        help="Add OpenAI ChatGPT models (gpt-3.5-turbo, gpt-4)",
    )
    parser.add_argument(
        "--add-claude",
        action="store_true",
        help="Add Anthropic's Claude models (claude-v1)",
    )
    parser.add_argument(
        "--add-bard",
        action="store_true",
        help="Add Google's Bard model",
    )
    parser.add_argument("--elo-results-file", type=str)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    set_global_vars(args.controller_url, args.moderate)
    set_global_vars_named(args.moderate)
    set_global_vars_anony(args.moderate)
    models = get_model_list(args.controller_url)

    demo = build_demo(models, args.elo_results_file)
    demo.queue(
        concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False
    ).launch(
        server_name=args.host, server_port=args.port, share=args.share, max_threads=200
    )
