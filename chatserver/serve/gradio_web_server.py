import argparse
from collections import defaultdict
import datetime
import json
import os
import time

import gradio as gr
import requests

from chatserver.conversation import (default_conversation, conv_templates,
    SeparatorStyle)
from chatserver.constants import LOGDIR
from chatserver.utils import build_logger, server_error_msg
from chatserver.serve.gradio_patch import Chatbot as grChatbot
from chatserver.serve.gradio_css import code_highlight_css


logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "ChatServer Client"}

upvote_msg = "👍  Upvote the last response"
downvote_msg = "👎  Downvote the last response"

priority = {
    "vicuna-13b": "aaaaaaa",
}


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


def load_demo(request: gr.Request):
    logger.info(f"load demo: {request.client.host}")
    state = default_conversation.copy()
    return (state,
            gr.Dropdown.update(visible=True),
            gr.Chatbot.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Row.update(visible=True),
            gr.Accordion.update(visible=True))


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load demo: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()
    return (state, gr.Dropdown.update(
               choices=models,
               value=models[0] if len(models) > 0 else ""),
            gr.Chatbot.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Row.update(visible=True),
            gr.Accordion.update(visible=True))


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    logger.info(f"vote_type: {vote_type}")
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, upvote_btn, downvote_btn, model_selector,
                         request: gr.Request):
    if len(state.messages) == state.offset:
        return upvote_btn, downvote_msg, ""
    if upvote_btn == "done":
        return "done", "done", ""
    vote_last_response(state, "upvote", model_selector, request)
    return "done", "done", ""


def downvote_last_response(state, upvote_btn, downvote_btn, model_selector,
                           request: gr.Request):
    if len(state.messages) == state.offset:
        return upvote_btn, downvote_msg, ""
    if upvote_btn == "done":
        return "done", "done", ""
    vote_last_response(state, "downvote", model_selector, request)
    return "done", "done", ""


def regenerate(state):
    if len(state.messages) == state.offset:
        # skip empty "Regenerate"
        return state, state.to_gradio_chatbot(), "", upvote_msg, downvote_msg

    state.messages[-1][-1] = None
    return state, state.to_gradio_chatbot(), "", upvote_msg, downvote_msg


def clear_history():
    state = default_conversation.copy()
    return state, state.to_gradio_chatbot(), ""


def add_text(state, text, request: gr.Request):
    text = text[:1536]  # Hard cut-off
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    return state, state.to_gradio_chatbot(), "", upvote_msg, downvote_msg


def http_bot(state, model_selector, temperature, max_new_tokens, request: gr.Request):
    start_tstamp = time.time()
    model_name = model_selector

    if len(state.messages) == state.offset:
        # Skip empty "Regenerate"
        yield state, state.to_gradio_chatbot()
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        if "bair-chat" in model_name: # Hardcode the condition
            template_name = "bair_v1"
        else:
            template_name = "v1"
        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address",
            json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield state, state.to_gradio_chatbot()
        return

    # Construct prompt
    prompt = state.get_prompt()

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep if state.sep_style == SeparatorStyle.SINGLE else state.sep2,
    }
    logger.info(f"==== request ====\n{pload}")

    state.messages[-1][-1] = "▌"
    yield state, state.to_gradio_chatbot()

    try:
        # Stream output
        response = requests.post(worker_addr + "/worker_generate_stream",
            headers=headers, json=pload, stream=True, timeout=10)
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt) + 2:]
                    state.messages[-1][-1] = output + "▌"
                    yield state, state.to_gradio_chatbot()
                else:
                    output = data["text"]
                    state.messages[-1][-1] = output + "▌"
                    yield state, state.to_gradio_chatbot()
                time.sleep(0.05)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        yield state, state.to_gradio_chatbot()
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield state, state.to_gradio_chatbot()

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(start_tstamp, 4),
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


notice_markdown = ("""
# Chat server\n
### Terms of Use\n
By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It does not provide safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.\n
### Choose a model to chat with
- [Vicuna](): a chat assistant fine-tuned from LLaMa on user-shared conversations. This one is expected to perform best according to our evaluation.
- [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html): a model fine-tuned from LLaMA on 52K instruction-following demonstrations.
- [LLaMa](https://arxiv.org/abs/2302.13971): open and efficient foundation language models
""")


learn_more_markdown = ("""
### Learn More
- Support this project by starting ChatServer on [Github](https://github.com/lm-sys/ChatServer).
- Read this blog [post]() about the Vicuna model.

### License
The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMa and [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI. Please contact us if you find any potential violation.
""")


css = code_highlight_css + """
pre {
    white-space: pre-wrap;       /* Since CSS 2.1 */
    white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
    white-space: -pre-wrap;      /* Opera 4-6 */
    white-space: -o-pre-wrap;    /* Opera 7 */
    word-wrap: break-word;       /* Internet Explorer 5.5+ */
}
"""


def build_demo():
    models = get_model_list()

    with gr.Blocks(title="Chat Server", theme=gr.themes.Base(), css=css) as demo:
        state = gr.State()

        # Draw layout
        notice = gr.Markdown(notice_markdown)

        with gr.Row(elem_id="model_selector_row"):
            model_selector = gr.Dropdown(
                choices=models,
                value=models[0] if len(models) > 0 else "",
                interactive=True,
                show_label=False).style(container=False)

        chatbot = grChatbot(elem_id="chatbot", visible=False).style(height=550)
        textbox = gr.Textbox(show_label=False,
            placeholder="Enter text and press ENTER", visible=False).style(container=False)

        with gr.Row(visible=False) as button_row:
            upvote_btn = gr.Button(value=upvote_msg)
            downvote_btn = gr.Button(value=downvote_msg)
            regenerate_btn = gr.Button(value="Regenerate")
            clear_btn = gr.Button(value="Clear history")

        with gr.Accordion("Parameters", open=False, visible=False) as parameter_row:
            temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Temperature",)
            max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens",)

        gr.Markdown(learn_more_markdown)

        # Register listeners
        upvote_btn.click(upvote_last_response,
            [state, upvote_btn, downvote_btn, model_selector],
            [upvote_btn, downvote_btn, textbox])
        downvote_btn.click(downvote_last_response,
            [state, upvote_btn, downvote_btn, model_selector],
            [upvote_btn, downvote_btn, textbox])
        regenerate_btn.click(regenerate, state,
            [state, chatbot, textbox, upvote_btn, downvote_btn]).then(
            http_bot, [state, model_selector, temperature, max_output_tokens],
            [state, chatbot])
        clear_btn.click(clear_history, None, [state, chatbot, textbox])

        textbox.submit(add_text, [state, textbox],
            [state, chatbot, textbox, upvote_btn, downvote_btn]).then(
            http_bot, [state, model_selector, temperature, max_output_tokens],
            [state, chatbot])

        if args.model_list_mode == "once":
            demo.load(load_demo, None, [state, model_selector,
                chatbot, textbox, button_row, parameter_row])
        elif args.model_list_mode == "reload":
            demo.load(load_demo_refresh_model_list, None, [state, model_selector,
                chatbot, textbox, button_row, parameter_row])
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=4)
    parser.add_argument("--model-list-mode", type=str, default="once",
        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_demo()
    demo.queue(concurrency_count=args.concurrency_count, status_update_rate=10,
               api_open=False).launch(
        server_name=args.host, server_port=args.port, share=args.share)
