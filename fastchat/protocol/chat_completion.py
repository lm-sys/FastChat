from typing import Optional, List, Dict, Any

import time

import shortuuid
from pydantic import BaseModel, Field


class ChatCompletionRequest(BaseModel):
    # TODO: support streaming, stop with a list of text etc.
    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 0.7
    n: int = 1
    max_tokens: Optional[int] = None
    stop: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=shortuuid.random)
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None
