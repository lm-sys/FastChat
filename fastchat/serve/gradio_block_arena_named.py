import json
import time

import gradio as gr
import numpy as np

from fastchat.conversation import get_default_conv_template
from fastchat.utils import (
    build_logger,
    violates_moderation,
    moderation_msg,
)
from fastchat.serve.gradio_patch import Chatbot as grChatbot
from fastchat.serve.gradio_web_server import (
    http_bot,
    get_conv_log_filename,
    no_change_btn,
    enable_btn,
    disable_btn,
)


logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")

num_models = 2
enable_moderation = False


def set_global_vars_named(enable_moderation_):
    global enable_moderation
    enable_moderation = enable_moderation_


def load_demo_side_by_side_named(models, url_params):
    states = (None,) * num_models

    model_left = models[0]
    if len(models) > 1:
        weights = ([8, 4, 2, 1] + [1] * 32)[:len(models) - 1]
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
    return ("",) + (disable_btn,) * 3


def rightvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"rightvote (named). ip: {request.client.host}")
    vote_last_response(
        [state0, state1], "rightvote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 3


def tievote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"tievote (named). ip: {request.client.host}")
    vote_last_response(
        [state0, state1], "tievote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 3


def regenerate(state0, state1, request: gr.Request):
    logger.info(f"regenerate (named). ip: {request.client.host}")
    states = [state0, state1]
    for i in range(num_models):
        states[i].messages[-1][-1] = None
        states[i].skip_next = False
    return states + [x.to_gradio_chatbot() for x in states] + [""] + [disable_btn] * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history (named). ip: {request.client.host}")
    return [None] * num_models + [None] * num_models + [""] + [disable_btn] * 5


def share_click(state0, state1, model_selector0, model_selector1,
                request: gr.Request):
    logger.info(f"share (named). ip: {request.client.host}")
    if state0 is not None and state1 is not None:
        vote_last_response(
            [state0, state1], "share", [model_selector0, model_selector1], request
        )


def add_text(state0, state1, text, request: gr.Request):
    logger.info(f"add_text (named). ip: {request.client.host}. len: {len(text)}")
    states = [state0, state1]

    for i in range(num_models):
        if states[i] is None:
            states[i] = get_default_conv_template("vicuna").copy()

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
            * 5
        )

    if enable_moderation:
        flagged = violates_moderation(text)
        if flagged:
            logger.info(f"violate moderation (named). ip: {request.client.host}. text: {text}")
            for i in range(num_models):
                states[i].skip_next = True
            return (
                states
                + [x.to_gradio_chatbot() for x in states]
                + [moderation_msg]
                + [
                    no_change_btn,
                ]
                * 5
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
        * 5
    )


def http_bot_all(
    state0,
    state1,
    model_selector0,
    model_selector1,
    temperature,
    max_new_tokens,
    request: gr.Request,
):
    logger.info(f"http_bot_all (named). ip: {request.client.host}")
    states = [state0, state1]
    model_selector = [model_selector0, model_selector1]
    gen = []
    for i in range(num_models):
        gen.append(
            http_bot(states[i], model_selector[i], temperature, max_new_tokens, request)
        )

    chatbots = [None] * num_models
    while True:
        stop = True
        for i in range(num_models):
            try:
                ret = next(gen[i])
                states[i], chatbots[i] = ret[0], ret[1]
                buttons = ret[2:]
                stop = False
            except StopIteration:
                pass
        yield states + chatbots + list(buttons)
        if stop:
            break

    for i in range(10):
        if i % 2 == 0:
            yield states + chatbots + [disable_btn] * 3 + list(buttons)[3:]
        else:
            yield states + chatbots + list(buttons)
        time.sleep(0.2)


def build_side_by_side_ui_named(models):
    notice_markdown = """
# ⚔️  Chatbot Arena ⚔️ 
Rules:
- Chat with two models side-by-side and vote for which one is better!
- You pick the models you want to chat with.
- You can continue chating and voting or click "Clear history" to start a new round.
- A leaderboard will be available soon.
- [[GitHub]](https://github.com/lm-sys/FastChat) [[Twitter]](https://twitter.com/lmsysorg) [[Discord]](https://discord.gg/h6kCZb72G7)

### Terms of use
By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. **The service collects user dialogue data for future research.**
The demo works better on desktop devices with a wide screen.

### Choose two models to chat with
| | |
| ---- | ---- |
| [Vicuna](https://vicuna.lmsys.org): a chat assistant fine-tuned from LLaMA on user-shared conversations by LMSYS. | [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/): a dialogue model for academic research by BAIR |
| [OpenAssistant (oasst)](https://open-assistant.io/): a chat-based assistant for everyone by LAION. | [Dolly](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm): an instruction-tuned open large language model by Databricks. |
| [ChatGLM](https://chatglm.cn/blog): an open bilingual dialogue language model by Tsinghua University | [StableLM](https://github.com/stability-AI/stableLM/): Stability AI language models. |
| [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html): a model fine-tuned from LLaMA on instruction-following demonstrations by Stanford. | [LLaMA](https://arxiv.org/abs/2302.13971): open and efficient foundation language models by Meta. |
"""

    learn_more_markdown = """
### License
The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
"""

    states = [gr.State() for _ in range(num_models)]
    model_selectors = [None] * num_models
    chatbots = [None] * num_models

    notice = gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Box(elem_id="share-region"):
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
                    chatbots[i] = grChatbot(label=label, elem_id=f"chatbot{i}",
                        visible=False).style(height=550)

        with gr.Box() as button_row:
            with gr.Row():
                leftvote_btn = gr.Button(value="👈  A is better", interactive=False)
                tie_btn = gr.Button(value="🤝  Tie", interactive=False)
                rightvote_btn = gr.Button(value="👉  B is better", interactive=False)

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
        max_output_tokens = gr.Slider(
            minimum=0,
            maximum=1024,
            value=512,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    gr.Markdown(learn_more_markdown)

    # Register listeners
    btn_list = [leftvote_btn, rightvote_btn, tie_btn, regenerate_btn, clear_btn]
    leftvote_btn.click(
        leftvote_last_response,
        states + model_selectors,
        [textbox, leftvote_btn, rightvote_btn, tie_btn],
    )
    rightvote_btn.click(
        rightvote_last_response,
        states + model_selectors,
        [textbox, leftvote_btn, rightvote_btn, tie_btn],
    )
    tie_btn.click(
        tievote_last_response,
        states + model_selectors,
        [textbox, leftvote_btn, rightvote_btn, tie_btn],
    )
    regenerate_btn.click(
        regenerate, states, states + chatbots + [textbox] + btn_list
    ).then(
        http_bot_all,
        states + model_selectors + [temperature, max_output_tokens],
        states + chatbots + btn_list,
    )
    clear_btn.click(clear_history, None, states + chatbots + [textbox] + btn_list)

    share_js="""
function (a, b, c, d) {
    const captureElement = document.querySelector('#share-region');
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
        add_text, states + [textbox], states + chatbots + [textbox] + btn_list
    ).then(
        http_bot_all,
        states + model_selectors + [temperature, max_output_tokens],
        states + chatbots + btn_list,
    )
    send_btn.click(
        add_text, states + [textbox], states + chatbots + [textbox] + btn_list
    ).then(
        http_bot_all,
        states + model_selectors + [temperature, max_output_tokens],
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

