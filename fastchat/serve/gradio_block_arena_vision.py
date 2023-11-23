"""
The gradio demo server for chatting with a single model.
"""

import argparse
from collections import defaultdict
import datetime
import json
import os
import random
import time
import uuid

import gradio as gr
import requests

from fastchat.conversation import SeparatorStyle
from fastchat.constants import (
    LOGDIR,
    WORKER_API_TIMEOUT,
    ErrorCode,
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    SERVER_ERROR_MSG,
    INACTIVE_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
    SESSION_EXPIRATION_TIME,
)
from fastchat.serve.gradio_web_server import (
    State,
    set_global_vars,
    get_conv_log_filename,
    get_model_list,
    load_demo_single,
    load_demo,
    upvote_last_response,
    downvote_last_response,
    flag_last_response,
    post_process_code,
    get_model_description_md,
    headers,
    no_change_btn,
    enable_btn,
    disable_btn,
    enable_moderation,
    acknowledgment_md,
    block_css,
    controller_url,
    bot_response
)
from fastchat.model.llava.constants import LLAVA_IMAGE_TOKEN
from fastchat.model.model_adapter import get_conversation_template
from fastchat.utils import (
    build_logger,
    oai_moderation,
    get_window_url_params_js,
    get_window_url_params_with_tos_js,
    parse_gradio_auth_creds,
)

logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")
enable_moderation = False

def regenerate(state, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.conv.update_last_message(None)
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = None
    return (state, [], "", None) + (disable_btn,) * 5


def add_text(state, model_selector, text, image, request: gr.Request):
    ip = request.client.host
    logger.info(f"add_text. ip: {ip}. len: {len(text)}")

    if state is None:
        state = State(model_selector)

    if len(text) <= 0:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 5

    if enable_moderation:
        flagged = oai_moderation(text)
        if flagged:
            logger.info(f"violate moderation. ip: {request.client.host}. text: {text}")
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), MODERATION_MSG, None) + (
                no_change_btn,
            ) * 5

    if image is not None and len(state.conv.get_images()) > 0:
        # reset convo with new image
        state.conv = get_conversation_template(state.model_name)

    conv = state.conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {request.client.host}. text: {text}")
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), CONVERSATION_LIMIT_MSG, None) + (
            no_change_btn,
        ) * 5

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off

    if image is not None:
        if "llava" in state.model_name.lower():
            if LLAVA_IMAGE_TOKEN not in text:
                text = text + "\n" + LLAVA_IMAGE_TOKEN

        text = (
            text,
            image,
        )

    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5

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