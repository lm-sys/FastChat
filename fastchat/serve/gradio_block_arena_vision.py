"""
The gradio demo server for chatting with a large multimodal model.

Usage:
python3 -m fastchat.serve.controller
python3 -m fastchat.serve.sglang_worker --model-path liuhaotian/llava-v1.5-7b --tokenizer-path llava-hf/llava-1.5-7b-hf
python3 -m fastchat.serve.gradio_web_server_multi --share --multimodal
"""

import os

import gradio as gr

from fastchat.serve.gradio_web_server import (
    upvote_last_response,
    downvote_last_response,
    flag_last_response,
    get_model_description_md,
    acknowledgment_md,
    bot_response,
    add_text,
    clear_history,
    regenerate,
)
from fastchat.utils import (
    build_logger,
)

logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")


def build_single_vision_language_model_ui(models, add_promotion_links=False):
    promotion = (
        """
| [GitHub](https://github.com/lm-sys/FastChat) | [Dataset](https://github.com/lm-sys/FastChat/blob/main/docs/dataset_release.md) | [Twitter](https://twitter.com/lmsysorg) | [Discord](https://discord.gg/HSWAKCrnFx) |
"""
        if add_promotion_links
        else ""
    )

    notice_markdown = f"""
# 🏔️ Chat with Open Large Vision-Language Models
{promotion}
"""

    state = gr.State()
    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Group():
        with gr.Row(elem_id="model_selector_row"):
            model_selector = gr.Dropdown(
                choices=models,
                value=models[0] if len(models) > 0 else "",
                interactive=True,
                show_label=False,
                container=False,
            )

        with gr.Accordion(
            f"🔍 Expand to see the descriptions of {len(models)} models", open=False
        ):
            model_description_md = get_model_description_md(models)
            gr.Markdown(model_description_md, elem_id="model_description_markdown")

    with gr.Row():
        with gr.Column(scale=3):
            textbox = gr.Textbox(
                show_label=False,
                placeholder="👉 Enter your prompt and press ENTER",
                container=False,
                render=False,
                elem_id="input_box",
            )
            imagebox = gr.Image(type="pil")

            cur_dir = os.path.dirname(os.path.abspath(__file__))

            with gr.Accordion("Parameters", open=False) as parameter_row:
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.2,
                    step=0.1,
                    interactive=True,
                    label="Temperature",
                )
                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    interactive=True,
                    label="Top P",
                )
                max_output_tokens = gr.Slider(
                    minimum=0,
                    maximum=1024,
                    value=512,
                    step=64,
                    interactive=True,
                    label="Max output tokens",
                )

            gr.Examples(
                examples=[
                    [
                        f"{cur_dir}/example_images/city.jpeg",
                        "What is unusual about this image?",
                    ],
                    [
                        f"{cur_dir}/example_images/fridge.jpeg",
                        "What is in this fridge?",
                    ],
                ],
                inputs=[imagebox, textbox],
            )

        with gr.Column(scale=8):
            chatbot = gr.Chatbot(
                elem_id="chatbot", label="Scroll down and start chatting", height=550
            )

            with gr.Row():
                with gr.Column(scale=8):
                    textbox.render()
                with gr.Column(scale=1, min_width=50):
                    send_btn = gr.Button(value="Send", variant="primary")
            with gr.Row(elem_id="buttons"):
                upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                clear_btn = gr.Button(value="🗑️  Clear", interactive=False)

    if add_promotion_links:
        gr.Markdown(acknowledgment_md, elem_id="ack_markdown")

    # Register listeners
    btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
    upvote_btn.click(
        upvote_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    downvote_btn.click(
        downvote_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    flag_btn.click(
        flag_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    regenerate_btn.click(
        regenerate, state, [state, chatbot, textbox, imagebox] + btn_list
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )
    clear_btn.click(clear_history, None, [state, chatbot, textbox, imagebox] + btn_list)

    model_selector.change(
        clear_history, None, [state, chatbot, textbox, imagebox] + btn_list
    )

    textbox.submit(
        add_text,
        [state, model_selector, textbox, imagebox],
        [state, chatbot, textbox, imagebox] + btn_list,
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )
    send_btn.click(
        add_text,
        [state, model_selector, textbox, imagebox],
        [state, chatbot, textbox, imagebox] + btn_list,
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )

    return [state, model_selector]
