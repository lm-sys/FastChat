"""
Chatbot Arena (battle) tab.
Users chat with two anonymous models.
"""

import json
import time

import gradio as gr
import numpy as np

from fastchat.constants import (
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    INACTIVE_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
)
from fastchat.model.model_adapter import get_conversation_template
from fastchat.serve.gradio_block_arena_named import flash_buttons
from fastchat.serve.gradio_web_server import (
    State,
    bot_response,
    get_conv_log_filename,
    no_change_btn,
    enable_btn,
    disable_btn,
    learn_more_md,
    ip_expiration_dict,
)
from fastchat.utils import (
    build_logger,
    violates_moderation,
)

logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")

num_sides = 2
enable_moderation = False
anony_names = ["", ""]
models = []


def set_global_vars_anony(enable_moderation_):
    global enable_moderation
    enable_moderation = enable_moderation_


def load_demo_side_by_side_anony(models_, url_params):
    global models
    models = models_

    states = (None,) * num_sides
    selector_updates = (
        gr.Markdown.update(visible=True),
        gr.Markdown.update(visible=True),
    )

    return (
        states
        + selector_updates
        + (gr.Chatbot.update(visible=True),) * num_sides
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
    for i in range(num_sides):
        states[i].conv.update_last_message(None)
    return states + [x.to_gradio_chatbot() for x in states] + [""] + [disable_btn] * 6


def clear_history(request: gr.Request):
    logger.info(f"clear_history (anony). ip: {request.client.host}")
    return (
        [None] * num_sides + [None] * num_sides + anony_names + [""] + [disable_btn] * 6
    )


def share_click(state0, state1, model_selector0, model_selector1, request: gr.Request):
    logger.info(f"share (anony). ip: {request.client.host}")
    if state0 is not None and state1 is not None:
        vote_last_response(
            [state0, state1], "share", [model_selector0, model_selector1], request
        )


SAMPLING_WEIGHTS = {
    "gpt-4": 1.5,
    "gpt-3.5-turbo": 1.5,
    "claude-v1": 1.5,
    "claude-instant-v1": 1.5,
    "palm-2": 1.5,
    "vicuna-33b": 1.5,
    "vicuna-13b": 1.5,
    "wizardlm-13b": 1.5,
    "gpt4all-13b-snoozy": 1.5,
    "guanaco-33b": 1.5,
    "koala-13b": 1.2,
    "vicuna-7b": 1.2,
    "mpt-7b-chat": 1.2,
    "oasst-pythia-12b": 1.2,
    "RWKV-4-Raven-14B": 1.2,
    "fastchat-t5-3b": 0.9,
    "alpaca-13b": 0.9,
    "chatglm-6b": 0.9,
    "chatglm2-6b": 0.9,
    "stablelm-tuned-alpha-7b": 0.3,
    "dolly-v2-12b": 0.3,
    "llama-13b": 0.1,
}

SAMPLING_BOOST_MODELS = []

model_pairs = []
model_pairs_weights = []


def add_text(
    state0, state1, model_selector0, model_selector1, text, request: gr.Request
):
    ip = request.client.host
    logger.info(f"add_text (anony). ip: {ip}. len: {len(text)}")
    states = [state0, state1]
    model_selectors = [model_selector0, model_selector1]

    # Init states if necessary
    if states[0] is None:
        assert states[1] is None
        global model_pairs, model_pairs_weights

        # Pick two models
        if len(model_pairs) == 0:
            for i in range(len(models)):
                for j in range(len(models)):
                    if i == j:
                        continue
                    a = models[i]
                    b = models[j]
                    w = SAMPLING_WEIGHTS.get(a, 1.0) * SAMPLING_WEIGHTS.get(b, 1.0)
                    if a in SAMPLING_BOOST_MODELS or b in SAMPLING_BOOST_MODELS:
                        w *= 5
                    model_pairs.append((a, b))
                    model_pairs_weights.append(w)

            model_pairs_weights = model_pairs_weights / np.sum(model_pairs_weights)
            # for p, w in zip(model_pairs, model_pairs_weights):
            #    print(p, w)

        if len(model_pairs) >= 1:
            idx = np.random.choice(len(model_pairs), p=model_pairs_weights)
            model_left, model_right = model_pairs[idx]
        else:
            model_left = model_right = models[0]

        states = [
            State(model_left),
            State(model_right),
        ]

    if len(text) <= 0:
        for i in range(num_sides):
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

    if ip_expiration_dict[ip] < time.time():
        logger.info(f"inactive (anony). ip: {request.client.host}. text: {text}")
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [INACTIVE_MSG]
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
            for i in range(num_sides):
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
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {request.client.host}. text: {text}")
        for i in range(num_sides):
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
    for i in range(num_sides):
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


def bot_response_multi(
    state0,
    state1,
    temperature,
    top_p,
    max_new_tokens,
    request: gr.Request,
):
    logger.info(f"bot_response_multi (anony). ip: {request.client.host}")

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
    for i in range(num_sides):
        gen.append(
            bot_response(
                states[i],
                temperature,
                top_p,
                max_new_tokens,
                request,
            )
        )

    chatbots = [None] * num_sides
    while True:
        stop = True
        for i in range(num_sides):
            try:
                ret = next(gen[i])
                states[i], chatbots[i] = ret[0], ret[1]
                stop = False
            except StopIteration:
                pass
        yield states + chatbots + [disable_btn] * 6
        if stop:
            break


def build_side_by_side_ui_anony(models):
    notice_markdown = """
# ⚔️  Chatbot Arena ⚔️ 
### Rules
- Chat with two anonymous models side-by-side and vote for which one is better!
- You can do multiple rounds of conversations before voting.
- The names of the models will be revealed after your vote. Conversations with identity keywords (e.g., ChatGPT, Bard, Vicuna) or any votes after the names are revealed will not count towards the leaderboard.
- Click "Clear history" to start a new round.
- | [Blog](https://lmsys.org/blog/2023-05-03-arena/) | [GitHub](https://github.com/lm-sys/FastChat) | [Paper](https://arxiv.org/abs/2306.05685) | [Twitter](https://twitter.com/lmsysorg) | [Discord](https://discord.gg/HSWAKCrnFx) |

### Terms of use
By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. **The service collects user dialogue data and reserves the right to distribute it under a Creative Commons Attribution (CC-BY) license.** The demo works better on desktop devices with a wide screen.

### Battle
Please scroll down and start chatting. You can view a leaderboard of participating models in the fourth tab above labeled 'Leaderboard' or by clicking [here](?leaderboard). The models include both closed-source models (e.g., ChatGPT) and open-source models (e.g., Vicuna).
"""

    states = [gr.State() for _ in range(num_sides)]
    model_selectors = [None] * num_sides
    chatbots = [None] * num_sides

    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Box(elem_id="share-region-anony"):
        with gr.Row():
            for i in range(num_sides):
                with gr.Column():
                    model_selectors[i] = gr.Markdown(anony_names[i])

        with gr.Row():
            for i in range(num_sides):
                label = "Model A" if i == 0 else "Model B"
                with gr.Column():
                    chatbots[i] = gr.Chatbot(
                        label=label, elem_id=f"chatbot", visible=False, height=550
                    )

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
                container=False,
            )
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
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
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
        add_text,
        states + model_selectors + [textbox],
        states + chatbots + [textbox] + btn_list,
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )

    send_btn.click(
        add_text,
        states + model_selectors + [textbox],
        states + chatbots + [textbox] + btn_list,
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
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
