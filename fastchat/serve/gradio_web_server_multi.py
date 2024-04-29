"""
The gradio demo server with multiple tabs.
It supports chatting with a single model or chatting with two models side-by-side.
"""

import argparse
import pickle
import time

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
from fastchat.serve.gradio_block_arena_vision import (
    build_single_vision_language_model_ui,
)
from fastchat.serve.gradio_block_arena_vision_anony import (
    build_side_by_side_vision_ui_anony,
    load_demo_side_by_side_vision_anony,
)
from fastchat.serve.gradio_block_arena_vision_named import (
    build_side_by_side_vision_ui_named,
)

from fastchat.serve.gradio_web_server import (
    set_global_vars,
    block_css,
    build_single_model_ui,
    build_about,
    get_model_list,
    load_demo_single,
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
    global models, all_models, vl_models

    ip = get_ip(request)
    logger.info(f"load_demo. ip: {ip}. params: {url_params}")

    selected = 0
    if "arena" in url_params:
        selected = 0
    elif "compare" in url_params:
        selected = 1
    elif "direct" in url_params or "model" in url_params:
        selected = 2
    elif "vision" in url_params:
        selected = 3
    elif "leaderboard" in url_params:
        selected = 4
    elif "about" in url_params:
        selected = 5

    if args.model_list_mode == "reload":
        models, all_models = get_model_list(
            args.controller_url,
            args.register_api_endpoint_file,
            vision_arena=False,
        )

        vl_models, all_vl_models = get_model_list(
            args.controller_url,
            args.register_api_endpoint_file,
            vision_arena=True,
        )

    single_updates = load_demo_single(models, url_params)
    side_by_side_anony_updates = load_demo_side_by_side_anony(all_models, url_params)
    side_by_side_named_updates = load_demo_side_by_side_named(models, url_params)

    vision_language_updates = load_demo_single(vl_models, url_params)
    side_by_side_vision_named_updates = load_demo_side_by_side_named(
        vl_models, url_params
    )
    side_by_side_vision_anony_updates = load_demo_side_by_side_vision_anony(
        vl_models, url_params
    )

    return (
        (gr.Tabs(selected=selected),)
        + single_updates
        + side_by_side_anony_updates
        + side_by_side_named_updates
        + side_by_side_vision_anony_updates
        + side_by_side_vision_named_updates
        + vision_language_updates
    )


def build_demo(models, vl_models, elo_results_file, leaderboard_table_file):
    text_size = gr.themes.sizes.text_md
    if args.show_terms_of_use:
        load_js = get_window_url_params_with_tos_js
    else:
        load_js = get_window_url_params_js

    head_js = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
"""
    if args.ga_id is not None:
        head_js += f"""
<script async src="https://www.googletagmanager.com/gtag/js?id={args.ga_id}"></script>
<script>
window.dataLayer = window.dataLayer || [];
function gtag(){{dataLayer.push(arguments);}}
gtag('js', new Date());

gtag('config', '{args.ga_id}');
window.__gradio_mode__ = "app";
</script>
        """

    with gr.Blocks(
        title="Chat with Open Large Language Models",
        theme=gr.themes.Default(text_size=text_size),
        css=block_css,
        head=head_js,
    ) as demo:
        with gr.Tabs() as tabs:
            with gr.Tab("Text Arena", id=0):
                with gr.Tab("⚔️  Arena (battle)", id=0):
                    side_by_side_anony_list = build_side_by_side_ui_anony(models)

                with gr.Tab("⚔️  Arena (side-by-side)", id=1):
                    side_by_side_named_list = build_side_by_side_ui_named(models)

                with gr.Tab("💬 Direct Chat", id=2):
                    single_model_list = build_single_model_ui(
                        models, add_promotion_links=True
                    )

            demo_tabs = (
                [tabs]
                + single_model_list
                + side_by_side_anony_list
                + side_by_side_named_list
            )

            if args.vision_arena:
                with gr.Tab("Vision Arena", id=3):
                    with gr.Tab("⚔️  Vision Arena (battle)", id=3):
                        side_by_side_vision_anony_list = (
                            build_side_by_side_vision_ui_anony(
                                vl_models,
                                random_questions=args.random_questions,
                            )
                        )

                    with gr.Tab("⚔️  Vision Arena (side-by-side)", id=4):
                        side_by_side_vision_named_list = (
                            build_side_by_side_vision_ui_named(
                                vl_models,
                                random_questions=args.random_questions,
                            )
                        )

                    with gr.Tab("👀 Vision Direct Chat", id=5):
                        single_vision_language_model_list = (
                            build_single_vision_language_model_ui(
                                vl_models,
                                add_promotion_links=True,
                                random_questions=args.random_questions,
                            )
                        )
                demo_tabs += (
                    side_by_side_vision_anony_list
                    + side_by_side_vision_named_list
                    + single_vision_language_model_list
                )

            if elo_results_file:
                with gr.Tab("Leaderboard", id=6):
                    build_leaderboard_tab(
                        elo_results_file, leaderboard_table_file, show_plot=True
                    )

            with gr.Tab("ℹ️  About Us", id=7):
                about = build_about()

        url_params = gr.JSON(visible=False)

        if args.model_list_mode not in ["once", "reload"]:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

        demo.load(
            load_demo,
            [url_params],
            demo_tabs,
            js=load_js,
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
        "--vision-arena", action="store_true", help="Show tabs for vision arena."
    )
    parser.add_argument(
        "--random-questions", type=str, help="Load random questions from a JSON file"
    )
    parser.add_argument(
        "--register-api-endpoint-file",
        type=str,
        help="Register API-based model endpoints from a JSON file",
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
    parser.add_argument(
        "--ga-id",
        type=str,
        help="the Google Analytics ID",
        default=None,
    )
    parser.add_argument(
        "--use-remote-storage",
        action="store_true",
        default=False,
        help="Uploads image files to google cloud storage if set to true",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # Set global variables
    set_global_vars(args.controller_url, args.moderate, args.use_remote_storage)
    set_global_vars_named(args.moderate)
    set_global_vars_anony(args.moderate)
    models, all_models = get_model_list(
        args.controller_url,
        args.register_api_endpoint_file,
        vision_arena=False,
    )

    vl_models, all_vl_models = get_model_list(
        args.controller_url,
        args.register_api_endpoint_file,
        vision_arena=True,
    )

    # Set authorization credentials
    auth = None
    if args.gradio_auth_path is not None:
        auth = parse_gradio_auth_creds(args.gradio_auth_path)

    # Launch the demo
    demo = build_demo(
        models,
        vl_models,
        args.elo_results_file,
        args.leaderboard_table_file,
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
        auth=auth,
        root_path=args.gradio_root_path,
        show_api=False,
    )
