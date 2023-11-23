"""
The gradio demo server for chatting with a single model.
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
enable_moderation = False


def build_single_vision_language_model_ui(models, add_promotion_links=False):
    promotion = (
        """
- | [GitHub](https://github.com/lm-sys/FastChat) | [Dataset](https://github.com/lm-sys/FastChat/blob/main/docs/dataset_release.md) | [Twitter](https://twitter.com/lmsysorg) | [Discord](https://discord.gg/HSWAKCrnFx) |
- Introducing Llama 2: The Next Generation Open Source Large Language Model. [[Website]](https://ai.meta.com/llama/)
- Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality. [[Blog]](https://lmsys.org/blog/2023-03-30-vicuna/)
"""
        if add_promotion_links
        else ""
    )

    notice_markdown = f"""
# 🏔️ Chat with Open Large Vision-Language Models
{promotion}

### Choose a model to chat with
"""

    state = gr.State()
    model_description_md = get_model_description_md(models)
    gr.Markdown(notice_markdown + model_description_md, elem_id="notice_markdown")

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row(elem_id="model_selector_row"):
                model_selector = gr.Dropdown(
                    choices=models,
                    value=models[0] if len(models) > 0 else "",
                    interactive=True,
                    show_label=False,
                    container=False,
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

        with gr.Column(scale=8):
            chatbot = gr.Chatbot(
                elem_id="chatbot", label="Scroll down and start chatting", height=550
            )

            with gr.Row():
                with gr.Column(scale=8):
                    textbox = gr.Textbox(
                        show_label=False,
                        placeholder="Enter your prompt here and press ENTER",
                        container=False,
                        elem_id="input_box",
                    )
                with gr.Column(scale=1, min_width=50):
                    send_btn = gr.Button(value="Send", variant="primary")
            with gr.Row(elem_id="buttons"):
                upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                clear_btn = gr.Button(value="🗑️  Clear", interactive=False)

            gr.Examples(
                examples=[
                    [
                        f"{cur_dir}/example_images/dog.jpeg",
                        "What animal is in this photo?",
                    ],
                    [
                        f"{cur_dir}/example_images/sunset.jpg",
                        "Where was this picture taken?",
                    ],
                ],
                inputs=[imagebox, textbox],
            )

    if add_promotion_links:
        gr.Markdown(acknowledgment_md)

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
