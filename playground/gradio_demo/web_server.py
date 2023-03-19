import argparse
import json
import time

import gradio as gr
import requests


def add_text(history, text):
    history = history + [[text, None]]
    return history, ""


def clear_history(history):
    return []


def dummy_bot(history):
    bot_response = "Today is a wonderful day"
    tmp = ""
    for word in bot_response.split(" "):
        time.sleep(1)
        tmp += word + " "
        history[-1][-1] = tmp
        yield history


def http_bot(history):
    stop_str = "\n"

    context = (
        f"A chat between a curious human and a knowledgeable artificial intelligence assistant.{stop_str}"
        f"Human: Hello! What can you do?{stop_str}"
        f"Assistant: As an AI assistant, I can answer questions and chat with you.{stop_str}"
        f"Human: What is the name of the tallest mountain in the world?{stop_str}"
        f"Assistant: Everest.{stop_str}"
    )
    for msg_pair in history:
        context += f"Human: {msg_pair[0]}{stop_str}"
        if msg_pair[1]:
            context += f"Assistant: {msg_pair[1]}{stop_str}"
        else:
            context += f"Assistant:"

    print(f"==== context ====\n{context}\n====")

    model = "default"
    prompt = context
    min_tokens = None
    top_p = 1
    echo = True

    headers = {"User-Agent": "Alpa Client"}
    pload = {
        "model": model,
        "prompt": prompt,
        "max_new_tokens": 64,
        "temperature": 0.8,
        "stop": stop_str,
    }
    response = requests.post(args.model_url, headers=headers, json=pload, stream=True)

    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"].split(f"{stop_str}Assistant: ")[-1]
            history[-1][-1] = output
            yield history


def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown(
            "# Chat server\n"
            "**Note**: This model lacks safety measures and may produce offensive content.\n"
        )
        chatbot = gr.Chatbot()
        textbox = gr.Textbox(show_label=False,
            placeholder="Enter text and press ENTER",).style(container=False)

        with gr.Row():
            upvote_btn = gr.Button(value="Upvote the last response")
            downvote_btn = gr.Button(value="Downvote the last response")
            clear_btn = gr.Button(value="Clear History")

        clear_btn.click(clear_history, inputs=[chatbot], outputs=[chatbot])
        textbox.submit(add_text, [chatbot, textbox], [chatbot, textbox]).then(
            http_bot, chatbot, chatbot,
        )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--model-url", type=str, default="http://localhost:10002")
    args = parser.parse_args()

    demo = build_demo()
    demo.queue(concurrency_count=1, status_update_rate=10).launch(
        server_name=args.host, server_port=args.port)
