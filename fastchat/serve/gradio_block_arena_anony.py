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
    SLOW_MODEL_MSG,
    BLIND_MODE_INPUT_CHAR_LEN_LIMIT,
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
    invisible_btn,
    acknowledgment_md,
    get_ip,
    get_model_description_md,
    _prepare_text_with_image,
)
from fastchat.serve.remote_logger import get_remote_logger
from fastchat.utils import (
    build_logger,
    moderation_filter,
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
        gr.Markdown(visible=True),
        gr.Markdown(visible=True),
    )

    return states + selector_updates


def vote_last_response(states, vote_type, model_selectors, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "models": [x for x in model_selectors],
            "states": [x.dict() for x in states],
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
    get_remote_logger().log(data)

    if ":" not in model_selectors[0]:
        for i in range(5):
            names = (
                "### Model A: " + states[0].model_name,
                "### Model B: " + states[1].model_name,
            )
            yield names + ("",) + (disable_btn,) * 4
            time.sleep(0.1)
    else:
        names = (
            "### Model A: " + states[0].model_name,
            "### Model B: " + states[1].model_name,
        )
        yield names + ("",) + (disable_btn,) * 4


def leftvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"leftvote (anony). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], "leftvote", [model_selector0, model_selector1], request
    ):
        yield x


def rightvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"rightvote (anony). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], "rightvote", [model_selector0, model_selector1], request
    ):
        yield x


def tievote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"tievote (anony). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], "tievote", [model_selector0, model_selector1], request
    ):
        yield x


def bothbad_vote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"bothbad_vote (anony). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], "bothbad_vote", [model_selector0, model_selector1], request
    ):
        yield x


def regenerate(state0, state1, request: gr.Request):
    logger.info(f"regenerate (anony). ip: {get_ip(request)}")
    states = [state0, state1]
    if state0.regen_support and state1.regen_support:
        for i in range(num_sides):
            states[i].conv.update_last_message(None)
        return (
            states + [x.to_gradio_chatbot() for x in states] + [""] + [disable_btn] * 6
        )
    states[0].skip_next = True
    states[1].skip_next = True
    return states + [x.to_gradio_chatbot() for x in states] + [""] + [no_change_btn] * 6


def clear_history(request: gr.Request):
    logger.info(f"clear_history (anony). ip: {get_ip(request)}")
    return (
        [None] * num_sides
        + [None] * num_sides
        + anony_names
        + [""]
        + [invisible_btn] * 4
        + [disable_btn] * 2
        + [""]
    )


def share_click(state0, state1, model_selector0, model_selector1, request: gr.Request):
    logger.info(f"share (anony). ip: {get_ip(request)}")
    if state0 is not None and state1 is not None:
        vote_last_response(
            [state0, state1], "share", [model_selector0, model_selector1], request
        )


SAMPLING_WEIGHTS = {
    # tier 0
    "gpt-4-0314": 4,
    "gpt-4-0613": 4,
    "gpt-4-1106-preview": 2,
    "gpt-4-0125-preview": 4,
    "gpt-4-turbo-2024-04-09": 4,
    "gpt-3.5-turbo-0125": 2,
    "claude-3-opus-20240229": 4,
    "claude-3-sonnet-20240229": 4,
    "claude-3-haiku-20240307": 4,
    "claude-2.1": 1,
    "zephyr-orpo-141b-A35b-v0.1": 2,
    "dbrx-instruct": 1,
    "command-r-plus": 4,
    "command-r": 2,
    "reka-flash": 4,
    "reka-flash-online": 4,
    "qwen1.5-72b-chat": 2,
    "qwen1.5-32b-chat": 2,
    "qwen1.5-14b-chat": 2,
    "qwen1.5-7b-chat": 2,
    "gemma-1.1-7b-it": 2,
    "gemma-1.1-2b-it": 1,
    "mixtral-8x7b-instruct-v0.1": 4,
    "mistral-7b-instruct-v0.2": 2,
    "mistral-large-2402": 4,
    "mistral-medium": 2,
    "starling-lm-7b-beta": 2,
    # tier 1
    "deluxe-chat-v1.3": 2,
    "llama-2-70b-chat": 2,
    "llama-2-13b-chat": 1,
    "llama-2-7b-chat": 1,
    "vicuna-33b": 1,
    "vicuna-13b": 1,
    "yi-34b-chat": 1,
}

# target model sampling weights will be boosted.
BATTLE_TARGETS = {
    "gpt-4-turbo-2024-04-09": {
        "gpt-4-1106-preview",
        "gpt-4-0125-preview",
        "claude-3-opus-20240229",
        "gemini-pro-dev-api",
    },
    "gemini-pro-dev-api": {
        "gpt-4-turbo-2024-04-09",
        "claude-3-opus-20240229",
        "gpt-4-0125-preview",
        "claude-3-sonnet-20240229",
    },
    "reka-flash": {
        "qwen1.5-72b-chat",
        "claude-3-haiku-20240307",
        "command-r-plus",
        "command-r",
    },
    "reka-flash-online": {
        "qwen1.5-72b-chat",
        "claude-3-haiku-20240307",
        "command-r-plus",
        "command-r",
    },
    "deluxe-chat-v1.3": {
        "gpt-4-1106-preview",
        "gpt-4-0125-preview",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
    },
    "qwen1.5-32b-chat": {
        "gpt-3.5-turbo-0125",
        "gpt-4-0613",
        "gpt-4-0125-preview",
        "llama-2-70b-chat",
        "mixtral-8x7b-instruct-v0.1",
        "mistral-large-2402",
        "yi-34b-chat",
    },
    "qwen1.5-14b-chat": {
        "starling-lm-7b-alpha",
        "claude-3-haiku-20240307",
        "gpt-3.5-turbo-0125",
        "openchat-3.5-0106",
        "mixtral-8x7b-instruct-v0.1",
    },
    "mistral-large-2402": {
        "gpt-4-0125-preview",
        "gpt-4-0613",
        "mixtral-8x7b-instruct-v0.1",
        "mistral-medium",
        "mistral-next",
        "claude-3-sonnet-20240229",
    },
    "gemma-1.1-2b-it": {
        "gpt-3.5-turbo-0125",
        "mixtral-8x7b-instruct-v0.1",
        "starling-lm-7b-beta",
        "llama-2-7b-chat",
        "mistral-7b-instruct-v0.2",
        "gemma-1.1-7b-it",
    },
    "zephyr-orpo-141b-A35b-v0.1": {
        "qwen1.5-72b-chat",
        "mistral-large-2402",
        "command-r-plus",
        "claude-3-haiku-20240307",
    },
}

SAMPLING_BOOST_MODELS = []

# outage models won't be sampled.
OUTAGE_MODELS = []


def get_sample_weight(model, outage_models, sampling_weights, sampling_boost_models):
    if model in outage_models:
        return 0
    weight = sampling_weights.get(model, 0)
    if model in sampling_boost_models:
        weight *= 5
    return weight


def get_battle_pair(
    models, battle_targets, outage_models, sampling_weights, sampling_boost_models
):
    if len(models) == 1:
        return models[0], models[0]

    model_weights = []
    for model in models:
        weight = get_sample_weight(
            model, outage_models, sampling_weights, sampling_boost_models
        )
        model_weights.append(weight)
    total_weight = np.sum(model_weights)
    model_weights = model_weights / total_weight
    chosen_idx = np.random.choice(len(models), p=model_weights)
    chosen_model = models[chosen_idx]
    # for p, w in zip(models, model_weights):
    #     print(p, w)

    rival_models = []
    rival_weights = []
    for model in models:
        if model == chosen_model:
            continue
        weight = get_sample_weight(
            model, outage_models, sampling_weights, sampling_boost_models
        )
        if (
            weight != 0
            and chosen_model in battle_targets
            and model in battle_targets[chosen_model]
        ):
            # boost to 50% chance
            weight = total_weight / len(battle_targets[chosen_model])
        rival_models.append(model)
        rival_weights.append(weight)
    # for p, w in zip(rival_models, rival_weights):
    #     print(p, w)
    rival_weights = rival_weights / np.sum(rival_weights)
    rival_idx = np.random.choice(len(rival_models), p=rival_weights)
    rival_model = rival_models[rival_idx]

    swap = np.random.randint(2)
    if swap == 0:
        return chosen_model, rival_model
    else:
        return rival_model, chosen_model


def add_text(
    state0, state1, model_selector0, model_selector1, text, image, request: gr.Request
):
    ip = get_ip(request)
    logger.info(f"add_text (anony). ip: {ip}. len: {len(text)}")
    states = [state0, state1]
    model_selectors = [model_selector0, model_selector1]

    # Init states if necessary
    if states[0] is None:
        assert states[1] is None

        model_left, model_right = get_battle_pair(
            models,
            BATTLE_TARGETS,
            OUTAGE_MODELS,
            SAMPLING_WEIGHTS,
            SAMPLING_BOOST_MODELS,
        )
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
            + ["", None]
            + [
                no_change_btn,
            ]
            * 6
            + [""]
        )

    model_list = [states[i].model_name for i in range(num_sides)]
    # turn on moderation in battle mode
    all_conv_text_left = states[0].conv.get_prompt()
    all_conv_text_right = states[0].conv.get_prompt()
    all_conv_text = (
        all_conv_text_left[-1000:] + all_conv_text_right[-1000:] + "\nuser: " + text
    )
    flagged = moderation_filter(all_conv_text, model_list, do_moderation=True)
    if flagged:
        logger.info(f"violate moderation (anony). ip: {ip}. text: {text}")
        # overwrite the original text
        text = MODERATION_MSG

    conv = states[0].conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {get_ip(request)}. text: {text}")
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [CONVERSATION_LIMIT_MSG, None]
            + [
                no_change_btn,
            ]
            * 6
            + [""]
        )

    text = text[:BLIND_MODE_INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    for i in range(num_sides):
        post_processed_text = _prepare_text_with_image(
            states[i], text, image, csam_flag=False
        )
        states[i].conv.append_message(states[i].conv.roles[0], post_processed_text)
        states[i].conv.append_message(states[i].conv.roles[1], None)
        states[i].skip_next = False

    hint_msg = ""
    for i in range(num_sides):
        if "deluxe" in states[i].model_name:
            hint_msg = SLOW_MODEL_MSG
    return (
        states
        + [x.to_gradio_chatbot() for x in states]
        + ["", None]
        + [
            disable_btn,
        ]
        * 6
        + [hint_msg]
    )


def bot_response_multi(
    state0,
    state1,
    temperature,
    top_p,
    max_new_tokens,
    request: gr.Request,
):
    logger.info(f"bot_response_multi (anony). ip: {get_ip(request)}")

    if state0 is None or state0.skip_next:
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
                apply_rate_limit=False,
                use_recommended_config=True,
            )
        )

    is_stream_batch = []
    for i in range(num_sides):
        is_stream_batch.append(
            states[i].model_name
            in [
                "gemini-pro",
                "gemini-pro-dev-api",
                "gemini-1.0-pro-vision",
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemma-1.1-2b-it",
                "gemma-1.1-7b-it",
            ]
        )
    chatbots = [None] * num_sides
    iters = 0
    while True:
        stop = True
        iters += 1
        for i in range(num_sides):
            try:
                # yield gemini fewer times as its chunk size is larger
                # otherwise, gemini will stream too fast
                if not is_stream_batch[i] or (iters % 30 == 1 or iters < 3):
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
# ⚔️  LMSYS Chatbot Arena: Benchmarking LLMs in the Wild
- | [Blog](https://lmsys.org/blog/2023-05-03-arena/) | [GitHub](https://github.com/lm-sys/FastChat) | [Paper](https://arxiv.org/abs/2306.05685) | [Dataset](https://github.com/lm-sys/FastChat/blob/main/docs/dataset_release.md) | [Twitter](https://twitter.com/lmsysorg) | [Discord](https://discord.gg/HSWAKCrnFx) |

## 📜 Rules
- Ask any question to two anonymous models (e.g., ChatGPT, Claude, Llama) and vote for the better one!
- You can continue chatting until you identify a winner.
- Vote won't be counted if model identity is revealed during conversation.

## 🏆 LMSYS Arena [Leaderboard](https://leaderboard.lmsys.org)
We've collected **500K+** human votes to compute an LLM Elo leaderboard.
Find out who is the 🥇LLM Champion!

## 👇 Chat now!
"""

    states = [gr.State() for _ in range(num_sides)]
    model_selectors = [None] * num_sides
    chatbots = [None] * num_sides

    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Group(elem_id="share-region-anony"):
        with gr.Accordion(
            f"🔍 Expand to see the descriptions of {len(models)} models", open=False
        ):
            model_description_md = get_model_description_md(models)
            gr.Markdown(model_description_md, elem_id="model_description_markdown")
        with gr.Row():
            for i in range(num_sides):
                label = "Model A" if i == 0 else "Model B"
                with gr.Column():
                    chatbots[i] = gr.Chatbot(
                        label=label,
                        elem_id="chatbot",
                        height=550,
                        show_copy_button=True,
                    )

        with gr.Row():
            for i in range(num_sides):
                with gr.Column():
                    model_selectors[i] = gr.Markdown(
                        anony_names[i], elem_id="model_selector_md"
                    )
        with gr.Row():
            slow_warning = gr.Markdown("")

    with gr.Row():
        leftvote_btn = gr.Button(
            value="👈  A is better", visible=False, interactive=False
        )
        rightvote_btn = gr.Button(
            value="👉  B is better", visible=False, interactive=False
        )
        tie_btn = gr.Button(value="🤝  Tie", visible=False, interactive=False)
        bothbad_btn = gr.Button(
            value="👎  Both are bad", visible=False, interactive=False
        )

    with gr.Row():
        textbox = gr.Textbox(
            show_label=False,
            placeholder="👉 Enter your prompt and press ENTER",
            elem_id="input_box",
        )
        send_btn = gr.Button(value="Send", variant="primary", scale=0)

    with gr.Row() as button_row:
        clear_btn = gr.Button(value="🎲 New Round", interactive=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        share_btn = gr.Button(value="📷  Share")

    with gr.Accordion("Parameters", open=False, visible=False) as parameter_row:
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
            maximum=2048,
            value=1024,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    gr.Markdown(acknowledgment_md, elem_id="ack_markdown")

    imagebox = gr.State(None)
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
        clear_history,
        None,
        states + chatbots + model_selectors + [textbox] + btn_list + [slow_warning],
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
    share_btn.click(share_click, states + model_selectors, [], js=share_js)

    textbox.submit(
        add_text,
        states + model_selectors + [textbox, imagebox],
        states + chatbots + [textbox, imagebox] + btn_list + [slow_warning],
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons,
        [],
        btn_list,
    )

    send_btn.click(
        add_text,
        states + model_selectors + [textbox, imagebox],
        states + chatbots + [textbox, imagebox] + btn_list,
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )

    return states + model_selectors
