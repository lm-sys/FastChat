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


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=shortuuid.random)
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[Dict[str, int]] = None


class EmbeddingsRequest(BaseModel):
    model: str
    input: str


class EmbeddingsResponse(BaseModel):
    object: str = "lists"
    data: List[Dict[str, Any]]
    model: str
    usage: Optional[Dict[str, int]] = None


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    suffix: Optional[str] = None
    temperature: Optional[float] = 0.7
    n: int = 1
    max_tokens: int
    stop: Optional[str] = None
    stream: bool = False
    logprobs: Optional[int] = None
    echo: bool = False


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=shortuuid.random)
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None
