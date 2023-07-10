"""Call API providers."""

import os
import random
import time

from fastchat.utils import build_logger
from fastchat.constants import WORKER_API_TIMEOUT


logger = build_logger("gradio_web_server", "gradio_web_server.log")


def openai_api_stream_iter(model_name, messages, temperature, top_p, max_new_tokens):
    import openai

    # Make requests
    gen_params = {
        "model": model_name,
        "prompt": messages,
        "temperature": temperature,
        "top_p": top_p,
    }
    logger.info(f"==== request ====\n{gen_params}")

    res = openai.ChatCompletion.create(
        model=model_name, messages=messages, temperature=temperature, stream=True
    )
    text = ""
    for chunk in res:
        text += chunk["choices"][0]["delta"].get("content", "")
        data = {
            "text": text,
            "error_code": 0,
        }
        yield data


def anthropic_api_stream_iter(model_name, prompt, temperature, top_p, max_new_tokens):
    import anthropic

    c = anthropic.Client(os.environ["ANTHROPIC_API_KEY"])

    # Make requests
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
    }
    logger.info(f"==== request ====\n{gen_params}")

    res = c.completion_stream(
        prompt=prompt,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        max_tokens_to_sample=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        model=model_name,
        stream=True,
    )
    for chunk in res:
        data = {
            "text": chunk["completion"],
            "error_code": 0,
        }
        yield data


def bard_api_stream_iter(state):
    import requests

    # TODO: we will use the official PaLM 2 API sooner or later,
    # and we will update this function accordingly. So here we just hard code the
    # Bard worker address. It is going to be deprecated anyway.
    conv = state.conv

    # Make requests
    gen_params = {
        "model": "bard",
        "prompt": state.messages,
    }
    logger.info(f"==== request ====\n{gen_params}")

    response = requests.post(
        "http://localhost:18900/chat",
        json={
            "content": conv.messages[-2][-1],
            "state": state.bard_session_state,
        },
        stream=False,
        timeout=WORKER_API_TIMEOUT,
    )
    resp_json = response.json()
    state.bard_session_state = resp_json["state"]
    content = resp_json["content"]
    # The Bard Web API does not support streaming yet. Here we have to simulate
    # the streaming behavior by adding some time.sleep().
    pos = 0
    while pos < len(content):
        # This is a fancy way to simulate token generation latency combined
        # with a Poisson process.
        pos += random.randint(1, 5)
        time.sleep(random.expovariate(50))
        data = {
            "text": content[:pos],
            "error_code": 0,
        }
        yield data


def init_palm_chat(model_name):
    import vertexai  # pip3 install google-cloud-aiplatform
    from vertexai.preview.language_models import ChatModel

    project_id = os.environ["GCP_PROJECT_ID"]
    location = "us-central1"
    vertexai.init(project=project_id, location=location)

    chat_model = ChatModel.from_pretrained(model_name)
    chat = chat_model.start_chat(examples=[])
    return chat


def palm_api_stream_iter(chat, message, temperature, top_p, max_new_tokens):
    parameters = {
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_new_tokens,
    }
    gen_params = {
        "model": "palm-2",
        "prompt": message,
    }
    gen_params.update(parameters)
    logger.info(f"==== request ====\n{gen_params}")

    response = chat.send_message(message, **parameters)
    content = response.text

    pos = 0
    while pos < len(content):
        # This is a fancy way to simulate token generation latency combined
        # with a Poisson process.
        pos += random.randint(10, 20)
        time.sleep(random.expovariate(50))
        data = {
            "text": content[:pos],
            "error_code": 0,
        }
        yield data
