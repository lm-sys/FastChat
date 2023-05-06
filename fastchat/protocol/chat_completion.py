from typing import Literal, Optional, List, Dict, Any, Union

import time

import shortuuid
from pydantic import BaseModel, Field


class ChatCompletionRequest(BaseModel):
    # TODO: support streaming, stop with a list of text etc.
    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: int = 1
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]]

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[Dict[str, int]] = None

class DeltaMessage(BaseModel):
    content: Optional[str]

class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]

class ChatCompletionResponseStreamChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[ChatCompletionResponseStreamChoice]
    # Similar to OpenAI, the stream mode does not return usage information

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
    id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None
