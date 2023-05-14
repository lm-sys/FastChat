"""
Chatbot Arena (battle) tab.
Users chat with two anonymous models.
"""

import json
import time

import gradio as gr
import numpy as np

from fastchat.model.model_adapter import get_conversation_template
from fastchat.serve.gradio_patch import Chatbot as grChatbot
from fastchat.serve.gradio_web_server import (
    http_bot,
    get_conv_log_filename,
    no_change_btn,
    enable_btn,
    disable_btn,
    learn_more_md,
)
from fastchat.utils import (
    build_logger,
    violates_moderation,
    moderation_msg,
)

logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")

num_models = 2
enable_moderation = False
anony_names = ["", ""]
models = []


def set_global_vars_anony(enable_moderation_):
    global enable_moderation
    enable_moderation = enable_moderation_


def load_demo_side_by_side_anony(models_, url_params):
    global models
    models = models_

    states = (None,) * num_models
    selector_updates = (
        gr.Markdown.update(visible=True),
        gr.Markdown.update(visible=True),
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

    if ":" not in model_selectors[0]:
        for i in range(15):
            names = (
                "### Model A: " + states[0].model_name,
                "### Model B: " + states[1].model_name,
            )
            yield names + ("",) + (disable_btn,) * 4
            time.sleep(0.2)
    else:
        names = (
            "### Model A: " + states[0].model_name,
            "### Model B: " + states[1].model_name,
        )
        yield names + ("",) + (disable_btn,) * 4


def leftvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"leftvote (anony). ip: {request.client.host}")
    for x in vote_last_response(
        [state0, state1], "leftvote", [model_selector0, model_selector1], request
    ):
        yield x


def rightvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"rightvote (anony). ip: {request.client.host}")
    for x in vote_last_response(
        [state0, state1], "rightvote", [model_selector0, model_selector1], request
    ):
        yield x


def tievote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"tievote (anony). ip: {request.client.host}")
    for x in vote_last_response(
        [state0, state1], "tievote", [model_selector0, model_selector1], request
    ):
        yield x


def bothbad_vote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"bothbad_vote (anony). ip: {request.client.host}")
    for x in vote_last_response(
        [state0, state1], "bothbad_vote", [model_selector0, model_selector1], request
    ):
        yield x


def regenerate(state0, state1, request: gr.Request):
    logger.info(f"regenerate (anony). ip: {request.client.host}")
    states = [state0, state1]
    for i in range(num_models):
        states[i].messages[-1][-1] = None
        states[i].skip_next = False
    return states + [x.to_gradio_chatbot() for x in states] + [""] + [disable_btn] * 6


def clear_history(request: gr.Request):
    logger.info(f"clear_history (anony). ip: {request.client.host}")
    return (
        [None] * num_models
        + [None] * num_models
        + anony_names
        + [""]
        + [disable_btn] * 6
    )


def share_click(state0, state1, model_selector0, model_selector1, request: gr.Request):
    logger.info(f"share (anony). ip: {request.client.host}")
    if state0 is not None and state1 is not None:
        vote_last_response(
            [state0, state1], "share", [model_selector0, model_selector1], request
        )


DEFAULT_WEIGHTS = {
    "gpt-4": 1.5,
    "gpt-3.5-turbo": 1.5,
    "claude-v1": 1.5,
    "claude-instant-v1.1": 1.5,
    "bard": 1.5,
    "vicuna-13b": 1.5,
    "koala-13b": 1.5,
    "RWKV-4-Raven-14B": 1.2,
    "oasst-pythia-12b": 1.2,
    "mpt-7b-chat": 1.2,
    "fastchat-t5-3b": 1,
    "alpaca-13b": 1,
    "chatglm-6b": 1,
    "stablelm-tuned-alpha-7b": 0.5,
    "dolly-v2-12b": 0.5,
    "llama-13b": 0.1,
}


def add_text(state0, state1, text, request: gr.Request):
    logger.info(f"add_text (anony). ip: {request.client.host}. len: {len(text)}")
    states = [state0, state1]

    if states[0] is None:
        assert states[1] is None
        weights = [DEFAULT_WEIGHTS.get(m, 1.0) for m in models]
        if len(models) > 1:
            weights = weights / np.sum(weights)
            model_left, model_right = np.random.choice(
                models, size=(2,), p=weights, replace=False
            )
        else:
            model_left = model_right = models[0]

        states = [
            get_conversation_template("vicuna"),
            get_conversation_template("vicuna"),
        ]
        states[0].model_name = model_left
        states[1].model_name = model_right

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
                f"violate moderation (anony). ip: {request.client.host}. text: {text}"
            )
            for i in range(num_models):
                states[i].skip_next = True
            return (
                states
                + [x.to_gradio_chatbot() for x in states]
                + [moderation_msg]
                + [
                    no_change_btn,
                ]
                * 6
            )

    text = text[:1536]  # Hard cut-off
    for i in range(num_models):
        states[i].append_message(states[i].roles[0], text)
        states[i].append_message(states[i].roles[1], None)
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
    model_selector0,
    model_selector1,
    temperature,
    top_p,
    max_new_tokens,
    request: gr.Request,
):
    logger.info(f"http_bot_all (anony). ip: {request.client.host}")

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
    model_selector = [state0.model_name, state1.model_name]
    gen = []
    for i in range(num_models):
        gen.append(
            http_bot(
                states[i],
                model_selector[i],
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


def build_side_by_side_ui_anony(models):
    notice_markdown = """
# ⚔️  Chatbot Arena ⚔️ 
### Rules
- Chat with two anonymous models side-by-side and vote for which one is better!
- You can do multiple rounds of conversations before voting.
- The names of the models will be revealed after your vote.
- Click "Clear history" to start a new round.
- [[Blog](https://lmsys.org/blog/2023-05-03-arena/)] [[GitHub]](https://github.com/lm-sys/FastChat) [[Twitter]](https://twitter.com/lmsysorg) [[Discord]](https://discord.gg/h6kCZb72G7)

### Terms of use
By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. **The service collects user dialogue data and reserves the right to distribute it under a Creative Commons Attribution (CC-BY) license.** The demo works better on desktop devices with a wide screen.

### Battle
Please scroll down and start chatting. You can view a leaderboard of participating models in the fourth tab above labeled 'Leaderboard' or by clicking [here](?leaderboard). The models include both closed-source models (e.g., ChatGPT) and open-source models (e.g., Vicuna).
"""

    states = [gr.State() for _ in range(num_models)]
    model_selectors = [None] * num_models
    chatbots = [None] * num_models

    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Box(elem_id="share-region-anony"):
        with gr.Row():
            for i in range(num_models):
                with gr.Column():
                    model_selectors[i] = gr.Markdown(anony_names[i])

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
            minimum=0,
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
        model_selectors + [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    rightvote_btn.click(
        rightvote_last_response,
        states + model_selectors,
        model_selectors + [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    tie_btn.click(
        tievote_last_response,
        states + model_selectors,
        model_selectors + [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    bothbad_btn.click(
        bothbad_vote_last_response,
        states + model_selectors,
        model_selectors + [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    regenerate_btn.click(
        regenerate, states, states + chatbots + [textbox] + btn_list
    ).then(
        http_bot_all,
        states + model_selectors + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    )
    clear_btn.click(
        clear_history, None, states + chatbots + model_selectors + [textbox] + btn_list
    )

    share_js = """
function (a, b, c, d) {
    const captureElement = document.querySelector('#share-region-anony');
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

    textbox.submit(
        add_text, states + [textbox], states + chatbots + [textbox] + btn_list
    ).then(
        http_bot_all,
        states + model_selectors + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    )
    send_btn.click(
        add_text, states + [textbox], states + chatbots + [textbox] + btn_list
    ).then(
        http_bot_all,
        states + model_selectors + [temperature, top_p, max_output_tokens],
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
