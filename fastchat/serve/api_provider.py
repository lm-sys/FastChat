"""Call API providers."""

import os
import random
import time

from fastchat.utils import build_logger
from fastchat.constants import WORKER_API_TIMEOUT


logger = build_logger("gradio_web_server", "gradio_web_server.log")


def openai_api_stream_iter(
    model_name,
    messages,
    temperature,
    top_p,
    max_new_tokens,
    api_base=None,
    api_key=None,
):
    import openai

    is_azure = False
    if "azure" in model_name:
        is_azure = True
        openai.api_type = "azure"
        openai.api_version = "2023-07-01-preview"
    else:
        openai.api_type = "open_ai"
        openai.api_version = None

    openai.api_base = api_base or "https://api.openai.com/v1"
    openai.api_key = api_key or os.environ["OPENAI_API_KEY"]
    if model_name == "gpt-4-turbo":
        model_name = "gpt-4-1106-preview"

    # Make requests
    gen_params = {
        "model": model_name,
        "prompt": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
    logger.info(f"==== request ====\n{gen_params}")

    if is_azure:
        res = openai.ChatCompletion.create(
            engine=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            stream=True,
        )
    else:
        res = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            stream=True,
        )
    text = ""
    for chunk in res:
        if len(chunk["choices"]) > 0:
            text += chunk["choices"][0]["delta"].get("content", "")
            data = {
                "text": text,
                "error_code": 0,
            }
            yield data


def anthropic_api_stream_iter(model_name, prompt, temperature, top_p, max_new_tokens):
    import anthropic

    c = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Make requests
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
    logger.info(f"==== request ====\n{gen_params}")

    res = c.completions.create(
        prompt=prompt,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        max_tokens_to_sample=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        model=model_name,
        stream=True,
    )
    text = ""
    for chunk in res:
        text += chunk.completion
        data = {
            "text": text,
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
