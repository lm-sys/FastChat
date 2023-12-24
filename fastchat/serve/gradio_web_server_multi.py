"""
The gradio demo server with multiple tabs.
It supports chatting with a single model or chatting with two models side-by-side.
"""

import argparse
import pickle
import time

import gradio as gr

from fastchat.constants import (
    SESSION_EXPIRATION_TIME,
)
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
from fastchat.serve.gradio_web_server import (
    set_global_vars,
    block_css,
    build_single_model_ui,
    build_about,
    get_model_list,
    load_demo_single,
    ip_expiration_dict,
    get_ip,
)
from fastchat.serve.monitor.monitor import build_leaderboard_tab
from fastchat.utils import (
    build_logger,
    get_window_url_params_js,
    get_window_url_params_with_tos_js,
    parse_gradio_auth_creds,
)

logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")


def load_demo(url_params, request: gr.Request):
    global models

    ip = get_ip(request)
    logger.info(f"load_demo. ip: {ip}. params: {url_params}")
    ip_expiration_dict[ip] = time.time() + SESSION_EXPIRATION_TIME

    selected = 0
    if "arena" in url_params:
        selected = 0
    elif "compare" in url_params:
        selected = 1
    elif "single" in url_params:
        selected = 2
    elif "leaderboard" in url_params:
        selected = 3

    if args.model_list_mode == "reload":
        if args.anony_only_for_proprietary_model:
            models = get_model_list(
                args.controller_url,
                args.register_openai_compatible_models,
                False,
                False,
                False,
            )
        else:
            models = get_model_list(
                args.controller_url,
                args.register_openai_compatible_models,
                args.add_chatgpt,
                args.add_claude,
                args.add_palm,
            )

    single_updates = load_demo_single(models, url_params)

    models_anony = list(models)
    if args.anony_only_for_proprietary_model:
        # Only enable these models in anony battles.
        if args.add_chatgpt:
            models_anony += [
                "gpt-4-0314",
                "gpt-4-0613",
                "gpt-3.5-turbo-0613",
                "gpt-3.5-turbo-1106",
            ]
        if args.add_claude:
            models_anony += ["claude-2.1", "claude-2.0", "claude-1", "claude-instant-1"]
        if args.add_palm:
            models_anony += ["gemini-pro"]
    anony_only_models = [
        "claude-1",
        "gpt-4-0314",
        "gpt-4-0613",
    ]
    for mdl in anony_only_models:
        models_anony.append(mdl)
    models_anony = list(set(models_anony))

    side_by_side_anony_updates = load_demo_side_by_side_anony(models_anony, url_params)
    side_by_side_named_updates = load_demo_side_by_side_named(models, url_params)
    return (
        (gr.Tabs.update(selected=selected),)
        + single_updates
        + side_by_side_anony_updates
        + side_by_side_named_updates
    )


def build_demo(models, elo_results_file, leaderboard_table_file):
    text_size = gr.themes.sizes.text_md
    with gr.Blocks(
        title="Chat with Open Large Language Models",
        theme=gr.themes.Default(text_size=text_size),
        css=block_css,
    ) as demo:
        with gr.Tabs() as tabs:
            with gr.Tab("Arena (battle)", id=0):
                side_by_side_anony_list = build_side_by_side_ui_anony(models)

            with gr.Tab("Arena (side-by-side)", id=1):
                side_by_side_named_list = build_side_by_side_ui_named(models)

            with gr.Tab("Direct Chat", id=2):
                single_model_list = build_single_model_ui(
                    models, add_promotion_links=True
                )
            if elo_results_file:
                with gr.Tab("Leaderboard", id=3):
                    build_leaderboard_tab(elo_results_file, leaderboard_table_file)
            with gr.Tab("About Us", id=4):
                about = build_about()

        url_params = gr.JSON(visible=False)

        if args.model_list_mode not in ["once", "reload"]:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

        if args.show_terms_of_use:
            load_js = get_window_url_params_with_tos_js
        else:
            load_js = get_window_url_params_js

        demo.load(
            load_demo,
            [url_params],
            [tabs]
            + single_model_list
            + side_by_side_anony_list
            + side_by_side_named_list,
            _js=load_js,
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Whether to generate a public, shareable link",
    )
    parser.add_argument(
        "--controller-url",
        type=str,
        default="http://localhost:21001",
        help="The address of the controller",
    )
    parser.add_argument(
        "--concurrency-count",
        type=int,
        default=10,
        help="The concurrency count of the gradio queue",
    )
    parser.add_argument(
        "--model-list-mode",
        type=str,
        default="once",
        choices=["once", "reload"],
        help="Whether to load the model list once or reload the model list every time.",
    )
    parser.add_argument(
        "--moderate",
        action="store_true",
        help="Enable content moderation to block unsafe inputs",
    )
    parser.add_argument(
        "--show-terms-of-use",
        action="store_true",
        help="Shows term of use before loading the demo",
    )
    parser.add_argument(
        "--add-chatgpt",
        action="store_true",
        help="Add OpenAI's ChatGPT models (gpt-3.5-turbo, gpt-4)",
    )
    parser.add_argument(
        "--add-claude",
        action="store_true",
        help="Add Anthropic's Claude models (claude-2, claude-instant-1)",
    )
    parser.add_argument(
        "--add-palm",
        action="store_true",
        help="Add Google's PaLM model (PaLM 2 for Chat: chat-bison@001)",
    )
    parser.add_argument(
        "--anony-only-for-proprietary-model",
        action="store_true",
        help="Only add ChatGPT, Claude, Bard under anony battle tab",
    )
    parser.add_argument(
        "--register-openai-compatible-models",
        type=str,
        help="Register custom OpenAI API compatible models by loading them from a JSON file",
    )
    parser.add_argument(
        "--gradio-auth-path",
        type=str,
        help='Set the gradio authentication file path. The file should contain one or more user:password pairs in this format: "u1:p1,u2:p2,u3:p3"',
        default=None,
    )
    parser.add_argument(
        "--elo-results-file", type=str, help="Load leaderboard results and plots"
    )
    parser.add_argument(
        "--leaderboard-table-file", type=str, help="Load leaderboard results and plots"
    )
    parser.add_argument(
        "--gradio-root-path",
        type=str,
        help="Sets the gradio root path, eg /abc/def. Useful when running behind a reverse-proxy or at a custom URL path prefix",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # Set global variables
    set_global_vars(args.controller_url, args.moderate)
    set_global_vars_named(args.moderate)
    set_global_vars_anony(args.moderate)
    if args.anony_only_for_proprietary_model:
        models = get_model_list(
            args.controller_url,
            args.register_openai_compatible_models,
            False,
            False,
            False,
        )
    else:
        models = get_model_list(
            args.controller_url,
            args.register_openai_compatible_models,
            args.add_chatgpt,
            args.add_claude,
            args.add_palm,
        )

    # Set authorization credentials
    auth = None
    if args.gradio_auth_path is not None:
        auth = parse_gradio_auth_creds(args.gradio_auth_path)

    # Launch the demo
    demo = build_demo(models, args.elo_results_file, args.leaderboard_table_file)
    demo.queue(
        concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_threads=200,
        auth=auth,
        root_path=args.gradio_root_path,
    )
