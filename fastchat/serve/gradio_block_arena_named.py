"""
Chatbot Arena (side-by-side) tab.
Users chat with two chosen models.
"""

import json
import time

import gradio as gr
import numpy as np

from fastchat.constants import (
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_LEN_LIMIT,
)
from fastchat.model.model_adapter import get_conversation_template
from fastchat.serve.gradio_patch import Chatbot as grChatbot
from fastchat.serve.gradio_web_server import (
    State,
    http_bot,
    get_conv_log_filename,
    get_model_description_md,
    no_change_btn,
    enable_btn,
    disable_btn,
    learn_more_md,
)
from fastchat.utils import (
    build_logger,
    violates_moderation,
)


logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")

num_models = 2
enable_moderation = False


def set_global_vars_named(enable_moderation_):
    global enable_moderation
    enable_moderation = enable_moderation_


def load_demo_side_by_side_named(models, url_params):
    states = (None,) * num_models

    model_left = models[0] if len(models) > 0 else ""
    if len(models) > 1:
        weights = ([8, 4, 2, 1] + [1] * 32)[: len(models) - 1]
        weights = weights / np.sum(weights)
        model_right = np.random.choice(models[1:], p=weights)
    else:
        model_right = model_left

    selector_updates = (
        gr.Dropdown.update(model_left, visible=True),
        gr.Dropdown.update(model_right, visible=True),
    )

    return (
        states
        + selector_updates
        + (gr.Chatbot.update(visible=True),) * num_models
        + (
            gr.Textbox.update(visible=True),
            gr.Box.update(visible=True),
            gr.Row.update(visible=True),
            gr.Row.update(visible=True),
            gr.Accordion.update(visible=True),
        )
    )


def vote_last_response(states, vote_type, model_selectors, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "models": [x for x in model_selectors],
            "states": [x.dict() for x in states],
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def leftvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"leftvote (named). ip: {request.client.host}")
    vote_last_response(
        [state0, state1], "leftvote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 4


def rightvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"rightvote (named). ip: {request.client.host}")
    vote_last_response(
        [state0, state1], "rightvote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 4


def tievote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"tievote (named). ip: {request.client.host}")
    vote_last_response(
        [state0, state1], "tievote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 4


def bothbad_vote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"bothbad_vote (named). ip: {request.client.host}")
    vote_last_response(
        [state0, state1], "bothbad_vote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 4


def regenerate(state0, state1, request: gr.Request):
    logger.info(f"regenerate (named). ip: {request.client.host}")
    states = [state0, state1]
    for i in range(num_models):
        states[i].conv.update_last_message(None)
    return states + [x.to_gradio_chatbot() for x in states] + [""] + [disable_btn] * 6


def clear_history(request: gr.Request):
    logger.info(f"clear_history (named). ip: {request.client.host}")
    return [None] * num_models + [None] * num_models + [""] + [disable_btn] * 6


def share_click(state0, state1, model_selector0, model_selector1, request: gr.Request):
    logger.info(f"share (named). ip: {request.client.host}")
    if state0 is not None and state1 is not None:
        vote_last_response(
            [state0, state1], "share", [model_selector0, model_selector1], request
        )


def add_text(
    state0, state1, model_selector0, model_selector1, text, request: gr.Request
):
    logger.info(f"add_text (named). ip: {request.client.host}. len: {len(text)}")
    states = [state0, state1]
    model_selectors = [model_selector0, model_selector1]

    for i in range(num_models):
        if states[i] is None:
            states[i] = State(model_selectors[i])

    if len(text) <= 0:
        for i in range(num_models):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [""]
            + [
                no_change_btn,
            ]
            * 6
        )

    if enable_moderation:
        flagged = violates_moderation(text)
        if flagged:
            logger.info(
                f"violate moderation (named). ip: {request.client.host}. text: {text}"
            )
            for i in range(num_models):
                states[i].skip_next = True
            return (
                states
                + [x.to_gradio_chatbot() for x in states]
                + [MODERATION_MSG]
                + [
                    no_change_btn,
                ]
                * 6
            )

    conv = states[0].conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_LEN_LIMIT:
        logger.info(
            f"hit conversation length limit. ip: {request.client.host}. text: {text}"
        )
        for i in range(num_models):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [CONVERSATION_LIMIT_MSG]
            + [
                no_change_btn,
            ]
            * 6
        )

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    for i in range(num_models):
        states[i].conv.append_message(states[i].conv.roles[0], text)
        states[i].conv.append_message(states[i].conv.roles[1], None)
        states[i].skip_next = False

    return (
        states
        + [x.to_gradio_chatbot() for x in states]
        + [""]
        + [
            disable_btn,
        ]
        * 6
    )


def http_bot_all(
    state0,
    state1,
    temperature,
    top_p,
    max_new_tokens,
    request: gr.Request,
):
    logger.info(f"http_bot_all (named). ip: {request.client.host}")

    if state0.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (
            state0,
            state1,
            state0.to_gradio_chatbot(),
            state1.to_gradio_chatbot(),
        ) + (no_change_btn,) * 6
        return

    states = [state0, state1]
    gen = []
    for i in range(num_models):
        gen.append(
            http_bot(
                states[i],
                temperature,
                top_p,
                max_new_tokens,
                request,
            )
        )

    chatbots = [None] * num_models
    while True:
        stop = True
        for i in range(num_models):
            try:
                ret = next(gen[i])
                states[i], chatbots[i] = ret[0], ret[1]
                stop = False
            except StopIteration:
                pass
        yield states + chatbots + [disable_btn] * 6
        if stop:
            break

    for i in range(10):
        if i % 2 == 0:
            yield states + chatbots + [disable_btn] * 4 + [enable_btn] * 2
        else:
            yield states + chatbots + [enable_btn] * 6
        time.sleep(0.2)


def build_side_by_side_ui_named(models):
    notice_markdown = """
# ⚔️  Chatbot Arena ⚔️ 
### Rules
- Chat with two models side-by-side and vote for which one is better!
- You pick the models you want to chat with.
- You can do multiple rounds of conversations before voting.
- Click "Clear history" to start a new round.
- [[Blog](https://lmsys.org/blog/2023-05-03-arena/)] [[GitHub]](https://github.com/lm-sys/FastChat) [[Twitter]](https://twitter.com/lmsysorg) [[Discord]](https://discord.gg/KjdtsE9V)

### Terms of use
By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. **The service collects user dialogue data and reserves the right to distribute it under a Creative Commons Attribution (CC-BY) license.** The demo works better on desktop devices with a wide screen.

### Choose two models to chat with (view [leaderboard](?leaderboard))
"""

    states = [gr.State() for _ in range(num_models)]
    model_selectors = [None] * num_models
    chatbots = [None] * num_models

    model_description_md = get_model_description_md(models)
    notice = gr.Markdown(
        notice_markdown + model_description_md, elem_id="notice_markdown"
    )

    with gr.Box(elem_id="share-region-named"):
        with gr.Row():
            for i in range(num_models):
                with gr.Column():
                    model_selectors[i] = gr.Dropdown(
                        choices=models,
                        value=models[i] if len(models) > i else "",
                        interactive=True,
                        show_label=False,
                    ).style(container=False)

        with gr.Row():
            for i in range(num_models):
                label = "Model A" if i == 0 else "Model B"
                with gr.Column():
                    chatbots[i] = grChatbot(
                        label=label, elem_id=f"chatbot", visible=False
                    ).style(height=550)

        with gr.Box() as button_row:
            with gr.Row():
                leftvote_btn = gr.Button(value="👈  A is better", interactive=False)
                rightvote_btn = gr.Button(value="👉  B is better", interactive=False)
                tie_btn = gr.Button(value="🤝  Tie", interactive=False)
                bothbad_btn = gr.Button(value="👎  Both are bad", interactive=False)

    with gr.Row():
        with gr.Column(scale=20):
            textbox = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press ENTER",
                visible=False,
            ).style(container=False)
        with gr.Column(scale=1, min_width=50):
            send_btn = gr.Button(value="Send", visible=False)

    with gr.Row() as button_row2:
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)
        share_btn = gr.Button(value="📷  Share")

    with gr.Accordion("Parameters", open=False, visible=True) as parameter_row:
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
            value=1.0,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=1024,
            value=512,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    gr.Markdown(learn_more_md)

    # Register listeners
    btn_list = [
        leftvote_btn,
        rightvote_btn,
        tie_btn,
        bothbad_btn,
        regenerate_btn,
        clear_btn,
    ]
    leftvote_btn.click(
        leftvote_last_response,
        states + model_selectors,
        [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    rightvote_btn.click(
        rightvote_last_response,
        states + model_selectors,
        [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    tie_btn.click(
        tievote_last_response,
        states + model_selectors,
        [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    bothbad_btn.click(
        bothbad_vote_last_response,
        states + model_selectors,
        [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    regenerate_btn.click(
        regenerate, states, states + chatbots + [textbox] + btn_list
    ).then(
        http_bot_all,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    )
    clear_btn.click(clear_history, None, states + chatbots + [textbox] + btn_list)

    share_js = """
function (a, b, c, d) {
    const captureElement = document.querySelector('#share-region-named');
    html2canvas(captureElement)
        .then(canvas => {
            canvas.style.display = 'none'
            document.body.appendChild(canvas)
            return canvas
        })
        .then(canvas => {
            const image = canvas.toDataURL('image/png')
            const a = document.createElement('a')
            a.setAttribute('download', 'chatbot-arena.png')
            a.setAttribute('href', image)
            a.click()
            canvas.remove()
        });
    return [a, b, c, d];
}
"""
    share_btn.click(share_click, states + model_selectors, [], _js=share_js)

    for i in range(num_models):
        model_selectors[i].change(
            clear_history, None, states + chatbots + [textbox] + btn_list
        )

    textbox.submit(
        add_text,
        states + model_selectors + [textbox],
        states + chatbots + [textbox] + btn_list,
    ).then(
        http_bot_all,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    )
    send_btn.click(
        add_text,
        states + model_selectors + [textbox],
        states + chatbots + [textbox] + btn_list,
    ).then(
        http_bot_all,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    )

    return (
        states,
        model_selectors,
        chatbots,
        textbox,
        send_btn,
        button_row,
        button_row2,
        parameter_row,
    )
