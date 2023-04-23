from typing import Dict, List, Optional
import asyncio
import os

import httpx
from fastchat.protocol.chat_completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
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


class ChatCompletion:
    OBJECT_NAME = "chat.completions"

    @classmethod
    def create(cls, *args, **kwargs) -> ChatCompletionResponse:
        """Creates a new chat completion for the provided messages and parameters.

        See `acreate` for more details.
        """
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
    ) -> ChatCompletionResponse:
        """Creates a new chat completion for the provided messages and parameters."""
        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            temperature=temperature,
            n=n,
            max_tokens=max_tokens,
            stop=stop,
        )
        client = ChatCompletionClient(_BASE_URL)
        response = await client.request_completion(request, timeout=timeout)
        return response
