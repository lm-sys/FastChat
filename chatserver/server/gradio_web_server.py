import argparse
from collections import defaultdict
import json
import time

import gradio as gr
import requests

from chatserver.conversation import default_conversation


def add_text(history, text):
    history = history + [[text, None]]
    return history, ""


def clear_history(history):
    return []


def http_bot(history, model_selector):
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address",
            json={"model_name": model_selector})
    worker_addr = ret.json()["address"]
    print(f"worker_addr: {worker_addr}")

    if worker_addr == "":
        history[-1][-1] = "**NETWORK ERROR. PLEASE TRY AGAIN OR CHOOSE OTHER MODELS.**"
        yield history
        return

    conv = default_conversation.copy()
    conv.append_gradio_chatbot_history(history)
    prompt = conv.get_prompt()

    txt = prompt.replace(conv.sep, '\n')
    print(f"==== Conversation ====\n{txt}")

    headers = {"User-Agent": "Alpa Client"}
    pload = {
        "prompt": prompt,
        "max_new_tokens": 64,
        "temperature": 0.8,
        "stop": conv.sep,
    }
    response = requests.post(worker_addr + "/generate_stream",
        headers=headers, json=pload, stream=True)

    sep = f"{conv.sep}{conv.roles[1]}: "
    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"].split(sep)[-1]
            history[-1][-1] = output
            yield history

    print(f"{output}")


priority = defaultdict(lambda: 10, {
    "facebook/opt-350m": 9,
    "facebook/opt-6.7b": 8,
    "facebook/llama-7b": 7,
})


def build_demo(models):
    models.sort(key=lambda x: priority[x])
    css = """#model_selector_row {width: 300px;}"""

    with gr.Blocks(title="Chat Server", css=css) as demo:
        gr.Markdown(
            "# Chat server\n"
            "**Note**: This service lacks safety measures and may produce offensive content.\n"
        )

        with gr.Row(elem_id="model_selector_row"):
            model_selector = gr.Dropdown(models,
                value=models[0],
                interactive=True,
                label="Choose a model to chat with.")

        chatbot = gr.Chatbot()
        textbox = gr.Textbox(show_label=False,
            placeholder="Enter text and press ENTER",).style(container=False)

        with gr.Row():
            upvote_btn = gr.Button(value="Upvote the last response")
            downvote_btn = gr.Button(value="Downvote the last response")
            clear_btn = gr.Button(value="Clear History")

        clear_btn.click(clear_history, inputs=[chatbot], outputs=[chatbot])
        textbox.submit(add_text, [chatbot, textbox], [chatbot, textbox]).then(
            http_bot, [chatbot, model_selector], chatbot,
        )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=2)
    args = parser.parse_args()

    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    print(f"Models: {models}")

    demo = build_demo(models)
    demo.queue(concurrency_count=args.concurrency_count, status_update_rate=10).launch(
        server_name=args.host, server_port=args.port)
