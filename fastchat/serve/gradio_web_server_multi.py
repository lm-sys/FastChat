"""
The gradio demo server with multiple tabs.
It supports chatting with a single model or chatting with two models side-by-side.
"""

import argparse
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
    load_demo_side_by_side_vision_named,
)
from fastchat.serve.gradio_global_state import Context

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
    alert_js,
    parse_gradio_auth_creds,
)

logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")


def build_visualizer():
    visualizer_markdown = """
    # 🔍 Arena Visualizer
    Arena visualizer provides interactive tools to explore and draw insights from our leaderboard data. 
    """
    gr.Markdown(visualizer_markdown, elem_id="visualizer_markdown")
    with gr.Tabs():
        with gr.Tab("Topic Explorer", id=0):
            topic_markdown = """ 
            This tool provides an interactive way to explore how people are using Chatbot Arena. 
            Using *[topic clustering](https://github.com/MaartenGr/BERTopic)*, we organized user-submitted prompts from Arena battles into broad and specific categories. 
            Dive in to uncover insights about the distribution and themes of these prompts! """
            gr.Markdown(topic_markdown)
            expandText = (
                "👇 Expand to see detailed instructions on how to use the visualizer"
            )
            with gr.Accordion(expandText, open=False):
                instructions = """
                - Hover Over Segments: View the category name, the number of prompts, and their percentage.
                    - *On mobile devices*: Tap instead of hover.
                - Click to Explore: 
                    - Click on a main category to see its subcategories.
                    - Click on subcategories to see example prompts in the sidebar.
                - Undo and Reset: Click the center of the chart to return to the top level.

                Visualizer is created using Arena battle data collected from 2024/6 to 2024/8.
                """
                gr.Markdown(instructions)

            frame = """
                        <iframe class="visualizer" width="100%"
                                src="https://storage.googleapis.com/public-arena-no-cors/index.html">
                        </iframe>
                    """
            gr.HTML(frame)
        with gr.Tab("Price Explorer", id=1):
            price_markdown = """
            This scatterplot presents a selection of models from the Arena, plotting their score against their cost. Only models with publicly available pricing and parameter information are included, meaning models like Gemini's experimental models are not displayed. Feel free to view price sources or add pricing information [here](https://github.com/lmarena/arena-catalog/blob/main/data/scatterplot-data.json).
            """
            gr.Markdown(price_markdown)
            expandText = (
                "👇 Expand to see detailed instructions on how to use the scatterplot"
            )
            with gr.Accordion(expandText, open=False):
                instructions = """
                - Hover Over Points: View the model's arena score, cost, organization, and license.
                - Click on Points: Click on a point to visit the model's website.
                - Use the Legend: Click an organization name on the right to display its models. To compare models, click multiple organization names.
                - Select Category: Use the dropdown menu in the upper-right corner to select a category and view the arena scores for that category.
                """
                gr.Markdown(instructions)

            frame = """<object type="text/html" data="https://storage.googleapis.com/public-arena-no-cors/scatterplot.html" width="100%" class="visualizer"></object>"""

            gr.HTML(frame)


def load_demo(context: Context, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"load_demo. ip: {ip}. params: {request.query_params}")

    inner_selected = 0
    if "arena" in request.query_params:
        inner_selected = 0
    elif "vision" in request.query_params:
        inner_selected = 0
    elif "compare" in request.query_params:
        inner_selected = 1
    elif "direct" in request.query_params or "model" in request.query_params:
        inner_selected = 2
    elif "leaderboard" in request.query_params:
        inner_selected = 3
    elif "about" in request.query_params:
        inner_selected = 4

    if args.model_list_mode == "reload":
        context.text_models, context.all_text_models = get_model_list(
            args.controller_url,
            args.register_api_endpoint_file,
            vision_arena=False,
        )

        context.vision_models, context.all_vision_models = get_model_list(
            args.controller_url,
            args.register_api_endpoint_file,
            vision_arena=True,
        )

    # Text models
    if args.vision_arena:
        side_by_side_anony_updates = load_demo_side_by_side_vision_anony()

        side_by_side_named_updates = load_demo_side_by_side_vision_named(
            context,
        )

        direct_chat_updates = load_demo_single(context, request.query_params)
    else:
        direct_chat_updates = load_demo_single(context, request.query_params)
        side_by_side_anony_updates = load_demo_side_by_side_anony(
            context.all_text_models, request.query_params
        )
        side_by_side_named_updates = load_demo_side_by_side_named(
            context.text_models, request.query_params
        )

    tabs_list = (
        [gr.Tabs(selected=inner_selected)]
        + side_by_side_anony_updates
        + side_by_side_named_updates
        + direct_chat_updates
    )

    return tabs_list


def build_demo(
    context: Context, elo_results_file: str, leaderboard_table_file, arena_hard_table
):
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
    text_size = gr.themes.sizes.text_lg
    with gr.Blocks(
        title="Chatbot Arena (formerly LMSYS): Free AI Chat to Compare & Test Best AI Chatbots",
        theme=gr.themes.Default(text_size=text_size),
        css=block_css,
        head=head_js,
    ) as demo:
        with gr.Tabs() as inner_tabs:
            if args.vision_arena:
                with gr.Tab("⚔️ Arena (battle)", id=0) as arena_tab:
                    arena_tab.select(None, None, None, js=load_js)
                    side_by_side_anony_list = build_side_by_side_vision_ui_anony(
                        context,
                        random_questions=args.random_questions,
                    )
                with gr.Tab("⚔️ Arena (side-by-side)", id=1) as side_by_side_tab:
                    side_by_side_tab.select(None, None, None, js=alert_js)
                    side_by_side_named_list = build_side_by_side_vision_ui_named(
                        context, random_questions=args.random_questions
                    )

                with gr.Tab("💬 Direct Chat", id=2) as direct_tab:
                    direct_tab.select(None, None, None, js=alert_js)
                    single_model_list = build_single_vision_language_model_ui(
                        context,
                        add_promotion_links=True,
                        random_questions=args.random_questions,
                    )

            else:
                with gr.Tab("⚔️ Arena (battle)", id=0) as arena_tab:
                    arena_tab.select(None, None, None, js=load_js)
                    side_by_side_anony_list = build_side_by_side_ui_anony(
                        context.all_text_models
                    )

                with gr.Tab("⚔️ Arena (side-by-side)", id=1) as side_by_side_tab:
                    side_by_side_tab.select(None, None, None, js=alert_js)
                    side_by_side_named_list = build_side_by_side_ui_named(
                        context.text_models
                    )

                with gr.Tab("💬 Direct Chat", id=2) as direct_tab:
                    direct_tab.select(None, None, None, js=alert_js)
                    single_model_list = build_single_model_ui(
                        context.text_models, add_promotion_links=True
                    )

            demo_tabs = (
                [inner_tabs]
                + side_by_side_anony_list
                + side_by_side_named_list
                + single_model_list
            )

            if elo_results_file:
                with gr.Tab("🏆 Leaderboard", id=3):
                    build_leaderboard_tab(
                        elo_results_file,
                        leaderboard_table_file,
                        arena_hard_table,
                        show_plot=True,
                    )
            if args.show_visualizer:
                with gr.Tab("🔍 Arena Visualizer", id=5):
                    build_visualizer()

            with gr.Tab("ℹ️ About Us", id=4):
                build_about()

        context_state = gr.State(context)

        if args.model_list_mode not in ["once", "reload"]:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

        demo.load(
            load_demo,
            [context_state],
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
        help='Set the gradio authentication file path. The file should contain one or \
              more user:password pairs in this format: "u1:p1,u2:p2,u3:p3"',
        default=None,
    )
    parser.add_argument(
        "--elo-results-file", type=str, help="Load leaderboard results and plots"
    )
    parser.add_argument(
        "--leaderboard-table-file", type=str, help="Load leaderboard results and plots"
    )
    parser.add_argument(
        "--arena-hard-table", type=str, help="Load leaderboard results and plots"
    )
    parser.add_argument(
        "--gradio-root-path",
        type=str,
        help="Sets the gradio root path, eg /abc/def. Useful when running behind a \
              reverse-proxy or at a custom URL path prefix",
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
    parser.add_argument(
        "--password",
        type=str,
        help="Set the password for the gradio web server",
    )
    parser.add_argument(
        "--show-visualizer",
        action="store_true",
        default=False,
        help="Show the Data Visualizer tab",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # Set global variables
    set_global_vars(args.controller_url, args.moderate, args.use_remote_storage)
    set_global_vars_named(args.moderate)
    set_global_vars_anony(args.moderate)
    text_models, all_text_models = get_model_list(
        args.controller_url,
        args.register_api_endpoint_file,
        vision_arena=False,
    )

    vision_models, all_vision_models = get_model_list(
        args.controller_url,
        args.register_api_endpoint_file,
        vision_arena=True,
    )

    models = text_models + [
        model for model in vision_models if model not in text_models
    ]
    all_models = all_text_models + [
        model for model in all_vision_models if model not in all_text_models
    ]
    context = Context(
        text_models,
        all_text_models,
        vision_models,
        all_vision_models,
        models,
        all_models,
    )

    # Set authorization credentials
    auth = None
    if args.gradio_auth_path is not None:
        auth = parse_gradio_auth_creds(args.gradio_auth_path)

    # Launch the demo
    demo = build_demo(
        context,
        args.elo_results_file,
        args.leaderboard_table_file,
        args.arena_hard_table,
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
