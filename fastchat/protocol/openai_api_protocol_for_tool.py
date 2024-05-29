from typing import Literal, Optional, List, Dict, Any, Union

import shortuuid
import time
from pydantic import BaseModel, Field

from fastchat.protocol.openai_api_protocol import UsageInfo


class FunctionBase(BaseModel):
    name: str = Field(max_length=64, pattern=r"^[a-zA-Z0-9_]+$")


class FunctionCallsMessage(FunctionBase):
    arguments: str


class Function(FunctionBase):
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ToolBase(BaseModel):
    type: Literal["function"]


class ToolCallsMessage(ToolBase):
    id: str
    function: FunctionCallsMessage


class Tool(ToolBase):
    function: Function


class ToolChoices(ToolBase):
    function: FunctionBase


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "function"]
    content: str
    name: Optional[str] = None
    tool_call_id: str = None
    tool_calls: Optional[List[ToolCallsMessage]] = None
    function_calls: Optional[FunctionCallsMessage] = None


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object"] = "text"


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = -1
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    response_format: Optional[ResponseFormat] = None
    tools: Optional[List[Tool]] = None
    tool_choices: Optional[Union[str, ToolChoices]] = None
    function_call: Optional[Union[str, FunctionBase]] = None
    functions: Optional[List[Function]] = None


class ChatCompletionResponseMessage(BaseModel):
    content: Optional[str] = None
    role: Literal["assistant", "tool"] = "assistant"
    tool_calls: Optional[List[ToolCallsMessage]] = None
    function_call: Optional[FunctionCallsMessage] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatCompletionResponseMessage
    finish_reason: Optional[
        Literal["stop", "length", "tool_calls", "function_call"]
    ] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo
