"""
The gradio demo server for chatting with a large multimodal model.

Usage:
python3 -m fastchat.serve.controller
python3 -m fastchat.serve.sglang_worker --model-path liuhaotian/llava-v1.5-7b --tokenizer-path llava-hf/llava-1.5-7b-hf
python3 -m fastchat.serve.gradio_web_server_multi --share --vision-arena
"""

import json
import os
import time
from typing import List, Union

import gradio as gr
from gradio.data_classes import FileData
import numpy as np

from fastchat.constants import (
    TEXT_MODERATION_MSG,
    IMAGE_MODERATION_MSG,
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
    SURVEY_LINK,
)
from fastchat.model.model_adapter import (
    get_conversation_template,
)
from fastchat.serve.gradio_global_state import Context
from fastchat.serve.gradio_web_server import (
    get_model_description_md,
    acknowledgment_md,
    bot_response,
    get_ip,
    disable_btn,
    State,
    get_conv_log_filename,
    get_remote_logger,
)
from fastchat.serve.vision.image import ImageFormat, Image
from fastchat.utils import (
    build_logger,
    moderation_filter,
    image_moderation_filter,
)

logger = build_logger("gradio_web_server", "gradio_web_server.log")

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True, visible=True)
disable_btn = gr.Button(interactive=False)
invisible_btn = gr.Button(interactive=False, visible=False)
visible_image_column = gr.Image(visible=True)
invisible_image_column = gr.Image(visible=False)
enable_multimodal = gr.MultimodalTextbox(
    interactive=True, visible=True, placeholder="Enter your prompt or add image here"
)
invisible_text = gr.Textbox(visible=False, value="", interactive=False)
visible_text = gr.Textbox(
    visible=True,
    value="",
    interactive=True,
    placeholder="👉 Enter your prompt and press ENTER",
)
disable_multimodal = gr.MultimodalTextbox(visible=False, value=None, interactive=False)


def get_vqa_sample():
    random_sample = np.random.choice(vqa_samples)
    question, path = random_sample["question"], random_sample["path"]
    res = {"text": "", "files": [path]}
    return (res, path)


def set_visible_image(textbox):
    images = textbox["files"]
    if len(images) == 0:
        return invisible_image_column
    elif len(images) > 1:
        gr.Warning(
            "We only support single image conversations. Please start a new round if you would like to chat using this image."
        )

    return visible_image_column


def set_invisible_image():
    return invisible_image_column


def add_image(textbox):
    images = textbox["files"]
    if len(images) == 0:
        return None

    return images[0]


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    filename = get_conv_log_filename(state.is_vision, state.has_csam_image)
    with open(filename, "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
    get_remote_logger().log(data)


def upvote_last_response(state, model_selector, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"upvote. ip: {ip}")
    vote_last_response(state, "upvote", model_selector, request)
    return (None,) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"downvote. ip: {ip}")
    vote_last_response(state, "downvote", model_selector, request)
    return (None,) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"flag. ip: {ip}")
    vote_last_response(state, "flag", model_selector, request)
    return (None,) + (disable_btn,) * 3


def regenerate(state, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"regenerate. ip: {ip}")
    if not state.regen_support:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 5
    state.conv.update_last_message(None)
    return (state, state.to_gradio_chatbot(), None) + (disable_btn,) * 5


def clear_history(request: gr.Request):
    ip = get_ip(request)
    logger.info(f"clear_history. ip: {ip}")
    state = None
    return (state, [], enable_multimodal, invisible_text, invisible_btn) + (
        disable_btn,
    ) * 5


def clear_history_example(request: gr.Request):
    ip = get_ip(request)
    logger.info(f"clear_history_example. ip: {ip}")
    state = None
    return (state, [], enable_multimodal, invisible_text, invisible_btn) + (
        disable_btn,
    ) * 5


# TODO(Chris): At some point, we would like this to be a live-reporting feature.
def report_csam_image(state, image):
    pass


def _prepare_text_with_image(state, text, images, csam_flag):
    if len(images) > 0:
        if len(state.conv.get_images()) > 0:
            # reset convo with new image
            state.conv = get_conversation_template(state.model_name)

        text = text, [images[0]]

    return text


# NOTE(chris): take multiple images later on
def convert_images_to_conversation_format(images):
    import base64

    MAX_NSFW_ENDPOINT_IMAGE_SIZE_IN_MB = 5 / 1.5
    conv_images = []
    if len(images) > 0:
        conv_image = Image(url=images[0])
        conv_image.to_conversation_format(MAX_NSFW_ENDPOINT_IMAGE_SIZE_IN_MB)
        conv_images.append(conv_image)

    return conv_images


def moderate_input(state, text, all_conv_text, model_list, images, ip):
    text_flagged = moderation_filter(all_conv_text, model_list)
    # flagged = moderation_filter(text, [state.model_name])
    nsfw_flagged, csam_flagged = False, False
    if len(images) > 0:
        nsfw_flagged, csam_flagged = image_moderation_filter(images[0])

    image_flagged = nsfw_flagged or csam_flagged
    if text_flagged or image_flagged:
        logger.info(f"violate moderation. ip: {ip}. text: {all_conv_text}")
        if text_flagged and not image_flagged:
            # overwrite the original text
            text = TEXT_MODERATION_MSG
        elif not text_flagged and image_flagged:
            text = IMAGE_MODERATION_MSG
        elif text_flagged and image_flagged:
            text = MODERATION_MSG

    if csam_flagged:
        state.has_csam_image = True
        report_csam_image(state, images[0])

    return text, image_flagged, csam_flagged


def add_text(
    state,
    model_selector,
    chat_input: Union[str, dict],
    context: Context,
    request: gr.Request,
):
    if isinstance(chat_input, dict):
        text, images = chat_input["text"], chat_input["files"]
    else:
        text, images = chat_input, []

    if (
        len(images) > 0
        and model_selector in context.text_models
        and model_selector not in context.vision_models
    ):
        gr.Warning(f"{model_selector} is a text-only model. Image is ignored.")
        images = []

    ip = get_ip(request)
    logger.info(f"add_text. ip: {ip}. len: {len(text)}")

    if state is None:
        if len(images) == 0:
            state = State(model_selector, is_vision=False)
        else:
            state = State(model_selector, is_vision=True)

    if len(text) <= 0:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), None, "", no_change_btn) + (
            no_change_btn,
        ) * 5

    all_conv_text = state.conv.get_prompt()
    all_conv_text = all_conv_text[-2000:] + "\nuser: " + text

    images = convert_images_to_conversation_format(images)

    text, image_flagged, csam_flag = moderate_input(
        state, text, all_conv_text, [state.model_name], images, ip
    )

    if image_flagged:
        logger.info(f"image flagged. ip: {ip}. text: {text}")
        state.skip_next = True
        return (
            state,
            state.to_gradio_chatbot(),
            {"text": IMAGE_MODERATION_MSG},
            "",
            no_change_btn,
        ) + (no_change_btn,) * 5

    if (len(state.conv.messages) - state.conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {ip}. text: {text}")
        state.skip_next = True
        return (
            state,
            state.to_gradio_chatbot(),
            {"text": CONVERSATION_LIMIT_MSG},
            "",
            no_change_btn,
        ) + (no_change_btn,) * 5

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    text = _prepare_text_with_image(state, text, images, csam_flag=csam_flag)
    state.conv.append_message(state.conv.roles[0], text)
    state.conv.append_message(state.conv.roles[1], None)
    return (
        state,
        state.to_gradio_chatbot(),
        disable_multimodal,
        visible_text,
        enable_btn,
    ) + (disable_btn,) * 5


def build_single_vision_language_model_ui(
    context: Context, add_promotion_links=False, random_questions=None
):
    promotion = (
        f"""
[Blog](https://blog.lmarena.ai/blog/2023/arena/) | [GitHub](https://github.com/lm-sys/FastChat) | [Paper](https://arxiv.org/abs/2403.04132) | [Dataset](https://github.com/lm-sys/FastChat/blob/main/docs/dataset_release.md) | [Twitter](https://twitter.com/lmsysorg) | [Discord](https://discord.gg/6GXcFg3TH8) | [Kaggle Competition](https://www.kaggle.com/competitions/lmsys-chatbot-arena)

{SURVEY_LINK}

**❗️ For research purposes, we log user prompts and images, and may release this data to the public in the future. Please do not upload any confidential or personal information.**

Note: You can only chat with <span style='color: #DE3163; font-weight: bold'>one image per conversation</span>. You can upload images less than 15MB. Click the "Random Example" button to chat with a random image."""
        if add_promotion_links
        else ""
    )

    notice_markdown = f"""
# 🏔️ Chatbot Arena (formerly LMSYS): Free AI Chat to Compare & Test Best AI Chatbots
{promotion}
"""

    state = gr.State()
    gr.Markdown(notice_markdown, elem_id="notice_markdown")
    vision_not_in_text_models = [
        model for model in context.vision_models if model not in context.text_models
    ]
    text_and_vision_models = context.text_models + vision_not_in_text_models
    context_state = gr.State(context)

    with gr.Group():
        with gr.Row(elem_id="model_selector_row"):
            model_selector = gr.Dropdown(
                choices=text_and_vision_models,
                value=text_and_vision_models[0]
                if len(text_and_vision_models) > 0
                else "",
                interactive=True,
                show_label=False,
                container=False,
            )

        with gr.Accordion(
            f"🔍 Expand to see the descriptions of {len(text_and_vision_models)} models",
            open=False,
        ):
            model_description_md = get_model_description_md(text_and_vision_models)
            gr.Markdown(model_description_md, elem_id="model_description_markdown")

    with gr.Row():
        with gr.Column(scale=2, visible=False) as image_column:
            imagebox = gr.Image(
                type="pil",
                show_label=False,
                interactive=False,
            )
        with gr.Column(scale=8):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="Scroll down and start chatting",
                height=650,
                show_copy_button=True,
            )

    with gr.Row():
        textbox = gr.Textbox(
            show_label=False,
            placeholder="👉 Enter your prompt and press ENTER",
            elem_id="input_box",
            visible=False,
        )

        send_btn = gr.Button(
            value="Send", variant="primary", scale=0, visible=False, interactive=False
        )

        multimodal_textbox = gr.MultimodalTextbox(
            file_types=["image"],
            show_label=False,
            placeholder="Enter your prompt or add image here",
            container=True,
            elem_id="input_box",
        )

    with gr.Row(elem_id="buttons"):
        if random_questions:
            global vqa_samples
            with open(random_questions, "r") as f:
                vqa_samples = json.load(f)
            random_btn = gr.Button(value="🎲 Random Example", interactive=True)
        upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
        downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
        flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        clear_btn = gr.Button(value="🗑️  Clear", interactive=False)

    with gr.Accordion("Parameters", open=False) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
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
            maximum=2048,
            value=1024,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

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
    regenerate_btn.click(regenerate, state, [state, chatbot, textbox] + btn_list).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )
    clear_btn.click(
        clear_history,
        None,
        [state, chatbot, multimodal_textbox, textbox, send_btn] + btn_list,
    )

    model_selector.change(
        clear_history,
        None,
        [state, chatbot, multimodal_textbox, textbox, send_btn] + btn_list,
    ).then(set_visible_image, [multimodal_textbox], [image_column])

    multimodal_textbox.input(add_image, [multimodal_textbox], [imagebox]).then(
        set_visible_image, [multimodal_textbox], [image_column]
    ).then(
        clear_history_example,
        None,
        [state, chatbot, multimodal_textbox, textbox, send_btn] + btn_list,
    )

    multimodal_textbox.submit(
        add_text,
        [state, model_selector, multimodal_textbox, context_state],
        [state, chatbot, multimodal_textbox, textbox, send_btn] + btn_list,
    ).then(set_invisible_image, [], [image_column]).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )

    textbox.submit(
        add_text,
        [state, model_selector, textbox, context_state],
        [state, chatbot, multimodal_textbox, textbox, send_btn] + btn_list,
    ).then(set_invisible_image, [], [image_column]).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )

    send_btn.click(
        add_text,
        [state, model_selector, textbox, context_state],
        [state, chatbot, multimodal_textbox, textbox, send_btn] + btn_list,
    ).then(set_invisible_image, [], [image_column]).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )

    if random_questions:
        random_btn.click(
            get_vqa_sample,  # First, get the VQA sample
            [],  # Pass the path to the VQA samples
            [multimodal_textbox, imagebox],  # Outputs are textbox and imagebox
        ).then(set_visible_image, [multimodal_textbox], [image_column]).then(
            clear_history_example,
            None,
            [state, chatbot, multimodal_textbox, textbox, send_btn] + btn_list,
        )

    return [state, model_selector]
