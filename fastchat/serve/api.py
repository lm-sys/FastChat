"""This module provides a ChatGPT-compatible Restful API for chat completion.

Usage:

python3 -m fastchat.serve.api

Reference: https://platform.openai.com/docs/api-reference/chat/create
"""
import argparse
import asyncio
import json
import logging
import os
from typing import Union, Dict, List, Any

import fastapi
import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseSettings

from fastchat.conversation import get_default_conv_template
from fastchat.protocol.chat_completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    DeltaMessage,
)

logger = logging.getLogger(__name__)

load_dotenv()


class AppSettings(BaseSettings):
    # The address of the model controller.
    FASTCHAT_CONTROLLER_URL: str = os.getenv(
        "FASTCHAT_CONTROLLER_URL", "http://localhost:21001"
    )


app_settings = AppSettings()
app = fastapi.FastAPI()
headers = {"User-Agent": "FastChat API Server"}


@app.get("/v1/models")
async def show_available_models():
    controller_url = app_settings.FASTCHAT_CONTROLLER_URL
    async with httpx.AsyncClient() as client:
        ret = await client.post(controller_url + "/refresh_all_workers")
        ret = await client.post(controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort()
    return {"data": [{"id": m} for m in models], "object": "list"}


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Creates a completion for the chat message"""
    gen_params = get_gen_params(
        request.model,
        request.messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        echo=False,
        stop=request.stop,
    )
    if request.stream:
        response_stream = chat_completion_stream(request.model, gen_params)
        return StreamingResponse(response_stream)

    choices = []
    # TODO: batch the requests. maybe not necessary if using CacheFlow worker
    chat_completions = []
    for i in range(request.n):
        content = asyncio.create_task(chat_completion(request.model, gen_params))
        chat_completions.append(content)

    for i, content_task in enumerate(chat_completions):
        content = await content_task
        choices.append(
            ChatCompletionResponseChoice(
                index=i,
                message=ChatMessage(role="assistant", content=content),
                # TODO: support other finish_reason
                finish_reason="stop",
            )
        )

    # TODO: support usage field
    # "usage": {
    #     "prompt_tokens": 9,
    #     "completion_tokens": 12,
    #     "total_tokens": 21
    # }
    return ChatCompletionResponse(choices=choices)


def get_gen_params(
    model_name: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    echo: bool,
    stop: Union[str, None],
):
    is_chatglm = "chatglm" in model_name.lower()
    # TODO(suquark): The template is currently a reference. Here we have to make a copy.
    conv = get_default_conv_template(model_name).copy()

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
        prompt = conv.messages[conv.offset :]
    else:
        prompt = conv.get_prompt()

    if max_tokens is None:
        max_tokens = 512

    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "max_new_tokens": max_tokens,
        "echo": echo,
        "stop": conv.stop_str,
        "stop_token_ids": conv.stop_token_ids,
    }
    logger.debug(f"==== request ====\n{gen_params}")
    return gen_params


async def _get_worker_address(model_name: str, client: httpx.AsyncClient) -> str:
    """
    Get worker address based on the requested model

    :param model_name: The worker's model name
    :param client: The httpx client to use
    :return: Worker address from the controller
    :raises: :class:`ValueError`: No available worker for requested model
    """
    controller_url = app_settings.FASTCHAT_CONTROLLER_URL

    ret = await client.post(
        controller_url + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]
    # No available worker
    if worker_addr == "":
        raise ValueError(f"No available worker for {model_name}")

    logger.debug(f"model_name: {model_name}, worker_addr: {worker_addr}")
    return worker_addr


async def chat_completion_stream(model_name: str, gen_params: Dict[str, Any]):
    async with httpx.AsyncClient() as client:
        worker_addr = await _get_worker_address(model_name, client)
        delimiter = "\0"

        async with client.stream(
            "POST",
            worker_addr + "/worker_generate_stream",
            headers=headers,
            json=gen_params,
            timeout=20,
        ) as response:
            i = 0
            async for chunk in response.aiter_text():
                if not chunk:
                    continue
                lines = chunk.split(delimiter)
                for line in lines:
                    if not line:
                        continue

                    data = json.loads(line)
                    if data["error_code"] == 0:
                        # print("*" * 30)
                        # print(line)  # TODO: Remove
                        # print("*" * 30)

                        output = data["text"].strip()
                        yield json.dumps(
                            ChatCompletionResponseStreamChoice(
                                index=i,
                                delta=DeltaMessage(content=output),
                                finish_reason="",
                            ).dict()
                        )
                        i += 1
            # Streaming the finsihg token
            yield json.dumps(
                ChatCompletionResponseStreamChoice(
                    index=i,
                    delta=DeltaMessage(content=""),
                    finish_reason="stop",
                ).dict()
            )


async def chat_completion(model_name: str, gen_params: Dict[str, Any]):
    async with httpx.AsyncClient() as client:
        worker_addr = await _get_worker_address(model_name, client)

        output = ""
        delimiter = b"\0"
        async with client.stream(
            "POST",
            worker_addr + "/worker_generate_stream",
            headers=headers,
            json=gen_params,
            timeout=20,
        ) as response:
            content = await response.aread()

        for chunk in content.split(delimiter):
            if not chunk:
                continue
            data = json.loads(chunk.decode())
            if data["error_code"] == 0:
                output = data["text"].strip()

        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FastChat ChatGPT-compatible Restful API server."
    )
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument(
        "--allow-credentials", action="store_true", help="allow credentials"
    )
    parser.add_argument(
        "--allowed-origins", type=json.loads, default=["*"], help="allowed origins"
    )
    parser.add_argument(
        "--allowed-methods", type=json.loads, default=["*"], help="allowed methods"
    )
    parser.add_argument(
        "--allowed-headers", type=json.loads, default=["*"], help="allowed headers"
    )

    args = parser.parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    logger.debug(f"==== args ====\n{args}")

    uvicorn.run("fastchat.serve.api:app", host=args.host, port=args.port, reload=True)
