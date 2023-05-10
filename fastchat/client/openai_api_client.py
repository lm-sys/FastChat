import asyncio
import json
import os
from typing import AsyncGenerator, Dict, List, Optional, Generator, Union

import httpx

from fastchat.utils import iter_over_async
from fastchat.protocol.openai_api_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
)


_BASE_URL = os.environ.get("FASTCHAT_API_BASE_URL", "http://localhost:8000")


def set_baseurl(base_url: str):
    global _BASE_URL
    _BASE_URL = base_url


class ChatCompletionClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def request_completion(
        self, request: ChatCompletionRequest, timeout: Optional[float] = None
    ) -> ChatCompletionResponse:
        """
        Create chat completion request
        :param request: The request data
        :param timeout: The timeout of the request
        :returns: Compleation stream
        """
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
    ) -> AsyncGenerator:
        """
        Create chat completion as a stream
        Parse the Event stream format: 
        https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
        :param request: The request data
        :param timeout: The timeout of the request
        :returns: Compleation stream
        """
        VALID_EVENT_STREAM_FIELD = ["id", "data", "event", "retry"]
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=request.dict(),
                timeout=timeout,
            ) as response:
                async for chunk in response.aiter_text():
                    if not chunk:
                        continue
                    for message in chunk.split("\n\n"):
                        if not message:
                            continue
                        lines = message.split("\n")

                        message_dict = {}
                        for line in lines:
                            colon_index = line.find(":")
                            if colon_index == 0 or colon_index == -1:
                                continue
                            message_key = line[:colon_index].strip()
                            message_value = line[colon_index + 1:]
                            if message_key in VALID_EVENT_STREAM_FIELD:
                                message_dict[message_key] = message_value
                        
                        data_field = message_dict["data"]
                        if data_field.strip() == "[DONE]":
                            break
                        yield ChatCompletionStreamResponse.parse_obj(json.loads(data_field))


class ChatCompletion:
    OBJECT_NAME = "chat.completions"

    @classmethod
    def create(cls, *args, **kwargs) -> Union[ChatCompletionResponse, Generator]:
        """Creates a new chat completion for the provided messages and parameters.

        See `acreate` for more details.
        """
        if kwargs.get("stream"):
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError as e:
                if str(e).startswith('There is no current event loop in thread'):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                else:
                    raise
            async_gen = cls.acreate(*args, **kwargs)
            async_gen_after_start = loop.run_until_complete(async_gen)
            sync_gen = iter_over_async(async_gen_after_start, loop)
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
