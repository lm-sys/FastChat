import asyncio
import json
import os
from typing import Dict, List, Optional, Generator, Union

import httpx

from fastchat.protocol.chat_completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChoice,
)

_BASE_URL = "http://localhost:8000"

if os.environ.get("FASTCHAT_BASE_URL"):
    _BASE_URL = os.environ.get("FASTCHAT_BASE_URL")


def set_baseurl(base_url: str):
    global _BASE_URL
    _BASE_URL = base_url


class ChatCompletionClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def request_completion(
        self, request: ChatCompletionRequest, timeout: Optional[float] = None
    ) -> ChatCompletionResponse:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=request.dict(),
                timeout=timeout,
            )
            response.raise_for_status()
            return ChatCompletionResponse.parse_obj(response.json())

    async def request_completion_stream(
        self, request: ChatCompletionRequest, timeout: Optional[float] = None
    ):
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=request.dict(),
                timeout=timeout,
            ) as response:
                async for chunk in response.aiter_lines():
                    if not chunk:
                        continue
                    try:
                        data = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    yield ChatCompletionResponseStreamChoice.parse_obj(data)


def iter_over_async(gen, loop):
    # ait = ait.__aiter__()
    while True:
        try:
            yield loop.run_until_complete(gen.__anext__())
        except StopAsyncIteration:
            break


class ChatCompletion:
    OBJECT_NAME = "chat.completions"

    @classmethod
    def create(cls, *args, **kwargs) -> Union[ChatCompletionResponse, Generator]:
        """Creates a new chat completion for the provided messages and parameters.

        See `acreate` for more details.
        """
        if kwargs.get("stream"):
            loop = asyncio.get_event_loop()
            async_gen = cls.acreate(*args, **kwargs)
            sync_gen = iter_over_async(async_gen, loop)
            return sync_gen
        else:
            return asyncio.run(cls.acreate(*args, **kwargs))

    @classmethod
    async def acreate(
        cls,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = 0.7,
        n: int = 1,
        max_tokens: Optional[int] = None,
        stop: Optional[str] = None,
        timeout: Optional[float] = None,
        stream: Optional[bool] = False,
    ):
        """Creates a new chat completion for the provided messages and parameters."""
        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            temperature=temperature,
            n=n,
            max_tokens=max_tokens,
            stop=stop,
            stream=stream,
        )
        client = ChatCompletionClient(_BASE_URL)
        if stream:
            return client.request_completion_stream(request, timeout=timeout)
        else:
            return await client.request_completion(request, timeout=timeout)
