"""This module provides a stream Restful API for chat completion stream.

Usage:

start server :
    python3 -m fastchat.serve.api_stream

test :
    curl http://localhost:8000/v1/chat/completions/stream   \
    -H "Content-Type: application/json"  \
    -d '{"model": "vicuna-7b-v1.1","messages": [{"role": "user", "content": "Hello!"}]}'

    Run curl at regular intervals until the returned result contains the stopword ([stop])
"""
import asyncio
from typing import Union, Dict, List, Any

import argparse
import json
import logging

import fastapi
import httpx
import uvicorn
from pydantic import BaseSettings
import datetime
import threading

from fastchat.protocol.chat_completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatCompletionResponseChoice,
)
from fastchat.conversation import get_default_conv_template, SeparatorStyle
from fastchat.serve.inference import compute_skip_echo_len

logger = logging.getLogger(__name__)


class AppSettings(BaseSettings):
    FASTCHAT_CONTROLLER_URL: str = "http://localhost:21001"


app_settings = AppSettings()
app = fastapi.FastAPI()
headers = {"User-Agent": "FastChat Stream API Server"}

stream_buffer = {}


def removeTimeoutBuffer():
    global stream_buffer
    for key in stream_buffer.copy():
        diff = datetime.datetime.now() - stream_buffer[key]["time"]
        seconds = diff.total_seconds()
        print(key + ": exists " + str(seconds) + " seconds")
        if seconds > 120:
            if stream_buffer[key]["stop"]:
                del stream_buffer[key]
                print(key + "：remove from cache")
            else:
                stream_buffer[key]["stop"] = True
                print(key + "：sign as stop")


@app.post("/v1/chat/completions/stream")
async def create_chat_completion_stream(request: ChatCompletionRequest):
    removeTimeoutBuffer()
    global stream_buffer
    msgid = request.messages[-1]["content"]
    now = datetime.datetime.now()
    # if cache is empty,then new thread to generate
    if stream_buffer.get(msgid) is None:
        stream_buffer[msgid] = {"response": "",
                                "stop": False, "history": request.messages, "time": now}
        sub_thread = threading.Thread(
            target=stream_item, args=(request.model, request.messages))
        sub_thread.start()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    response = stream_buffer[msgid]["response"]
    history = request.messages
    stream_buffer[msgid]["history"] = history
    # if generate finished , then return response + stopword
    if stream_buffer[msgid]["stop"]:
        stream_buffer[msgid]["history"].append(
            {"role": "assistant", "content": response})
        history = stream_buffer[msgid]["history"]
        response = response + '[stop]'
        log = time + " | INFO | stream | INFO: msgid:" + msgid + ', response:' + response
        print(log)
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    return answer


def stream_item(model, messages):
    request = ChatCompletionRequest(model=model, messages=messages)
    asyncio.run(create_chat_completion(request))


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Creates a completion for the chat message"""
    payload, skip_echo_len = generate_payload(
        request.model,
        request.messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stop=request.stop,
    )

    choices = []
    chat_completions = []
    for i in range(request.n):
        content = asyncio.create_task(chat_completion(
            request.model, payload, skip_echo_len, request.messages[-1]["content"]))
        chat_completions.append(content)

    for i, content_task in enumerate(chat_completions):
        content = await content_task
        choices.append(
            ChatCompletionResponseChoice(
                index=i,
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        )

    return ChatCompletionResponse(choices=choices)


def generate_payload(
    model_name: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    stop: Union[str, None],
):
    is_chatglm = "chatglm" in model_name.lower()
    conv = get_default_conv_template(model_name).copy()

    conv.messages = list(conv.messages)

    for message in messages:
        msg_role = message["role"]
        if msg_role == "system":
            conv.system = message["content"]
        elif msg_role == "user":
            conv.append_message(conv.roles[0], message["content"])
        elif msg_role == "assistant":
            conv.append_message(conv.roles[1], message["content"])
        else:
            raise ValueError(f"Unknown role: {msg_role}")

    # Add a blank message for the assistant.
    conv.append_message(conv.roles[1], None)

    if is_chatglm:
        prompt = conv.messages[conv.offset:]
    else:
        prompt = conv.get_prompt()
    skip_echo_len = compute_skip_echo_len(model_name, conv, prompt)

    if stop is None:
        stop = conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2

    if max_tokens is None:
        max_tokens = 512

    payload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "max_new_tokens": max_tokens,
        "stop": stop,
    }

    logger.debug(f"==== request ====\n{payload}")
    return payload, skip_echo_len


async def chat_completion(model_name: str, payload: Dict[str, Any], skip_echo_len: int, msgid: str):
    global stream_buffer
    controller_url = app_settings.FASTCHAT_CONTROLLER_URL
    async with httpx.AsyncClient() as client:
        ret = await client.post(
            controller_url + "/get_worker_address", json={"model": model_name}
        )
        worker_addr = ret.json()["address"]
        # No available worker
        if worker_addr == "":
            raise ValueError(f"No available worker for {model_name}")

        logger.debug(f"model_name: {model_name}, worker_addr: {worker_addr}")
        async with client.stream(
            "POST",
            worker_addr + "/worker_generate_stream",
            headers=headers,
            json=payload,
            timeout=120,
        ) as response:
            async for chunk in response.aiter_text():
                chunk = chunk.replace('\\ufffd', '').replace(
                    '\\uff0c', ',').replace('\\uff01', '!').replace('\\uff1f', '?')
                content = chunk.encode('utf-8').decode('unicode_escape')
                if content.strip() != "":
                    now = datetime.datetime.now()
                    b = content.rindex('ASSISTANT: ')
                    e = content.index('", "error_code"')
                    if (b * e) > 0:
                        output = content[b+11:e]
                        stream_buffer[msgid] = {
                            "response": output, "stop": False, "history": stream_buffer[msgid]["history"], "time": now}
        stream_buffer[msgid]["stop"] = True
        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FastChat Stream Restful API server."
    )
    parser.add_argument("--host", type=str,
                        default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")

    args = parser.parse_args()
    uvicorn.run("fastchat.serve.api_stream:app", host=args.host,
                port=args.port, reload=True)
