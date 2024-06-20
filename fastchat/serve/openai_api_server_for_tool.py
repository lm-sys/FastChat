"""A server that provides OpenAI-compatible RESTful APIs. It supports:

- Chat Completions. (Reference: https://platform.openai.com/docs/api-reference/chat)
- Completions. (Reference: https://platform.openai.com/docs/api-reference/completions)
- Embeddings. (Reference: https://platform.openai.com/docs/api-reference/embeddings)

Usage:
python3 -m fastchat.serve.openai_api_server
"""

import asyncio
import copy
import json
import os
import uuid
from datetime import datetime
from typing import Union, Optional, List, Dict, Any

import fastapi
from fastapi import HTTPException, Depends

import uvicorn
from fastapi.security import HTTPBearer
from starlette.responses import StreamingResponse

from fastchat.constants import ErrorCode
from fastchat.conversation import Conversation, SeparatorStyle
from fastchat.protocol.api_protocol import (
    APITokenCheckRequest,
    APIChatCompletionRequest,
)
from fastchat.protocol.openai_api_protocol import (
    EmbeddingsRequest,
    UsageInfo,
    CompletionRequest,
)
from fastchat.protocol.openai_api_protocol_for_tool import (
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatCompletionResponse,
    ChatMessage,
    ToolCallsMessage,
    FunctionCallsMessage,
    ChatCompletionResponseMessage,
    ToolChoice,
)
from fastchat.serve.openai_api_server import (
    AppSettings,
    check_api_key,
    create_chat_completion,
    create_completion,
    create_embeddings,
    count_tokens,
    create_openai_api_server,
    show_available_models,
    check_model,
    check_requests,
    get_worker_address,
    check_length,
    chat_completion_stream_generator,
    generate_completion,
    create_error_response,
    get_conv,
    _add_to_set,
)
from fastchat.utils import build_logger

logger = build_logger("openai_api_server_for_tool", "openai_api_server_for_tool.log")

ACTION_TOKEN = "Action:"
ARGS_TOKEN = "Action Input:"
OBSERVATION_TOKEN = "Observation:"
ANSWER_TOKEN = "Answer:"

TOOL_DESC = """{name}: {name} API。{description} 输入参数: {parameters} Format the arguments as a JSON object."""

TOOL_TEMPLATE = """# 基本信息
当前时间: {date}
# 工具

## 你拥有如下工具：

{tools_text}

## 当你需要调用工具时，请在你的回复中穿插如下的工具调用命令，可以根据需求调用零次或多次：

工具调用
Action: 工具的名称，必须是[{tools_name_text}]之一
Action Input: 工具的输入
Observation: <result>工具返回的结果</result>
Answer: 根据Observation总结本次工具调用返回的结果

"""
PROMPT_TEMPLATE = """# 指令

请注意：你具有工具调用能力，也具有运行代码的能力，不要在回复中说你做不到。
"""
SPECIAL_PREFIX_TEMPLATE_TOOL = "。你可以使用工具：[{tool_names}]"

SPECIAL_PREFIX_TEMPLATE_TOOL_FOR_CHAT = "。你必须使用工具：[{tool_names}]"

app_settings = AppSettings()
app = fastapi.FastAPI()
headers = {"User-Agent": "FastChat API Server"}
get_bearer_token = HTTPBearer(auto_error=False)


def add_extra_stop_words(stop_words):
    if stop_words:
        _stop_words = []
        _stop_words.extend(stop_words)
        for x in stop_words:
            s = x.lstrip("\n")
            if s and (s not in _stop_words):
                _stop_words.append(s)
        return _stop_words
    return stop_words


def parse_function_messages(request: ChatCompletionRequest) -> ChatCompletionRequest:
    messages = copy.deepcopy(request.messages)
    if request.tools:
        tools = [func.function for func in request.tools]
    else:
        tools = request.functions
    # 没有用户发出的消息，报错
    if all(m.role != "user" for m in messages):
        raise HTTPException(
            status_code=400,
            detail="Invalid request: Expecting at least one user message.",
        )
    # 如果请求体有 工具 调用 修改system prompt
    if tools:
        # add stop word
        stop_words = add_extra_stop_words(request.stop)
        stop_words = stop_words or []
        if "Observation:" not in stop_words:
            stop_words.append("Observation:")
        request.stop = stop_words
        # update system message
        tool_system = ""
        tools_text = []
        tools_name_text = []
        for func_info in tools:
            name = func_info.name
            description = func_info.description
            parameters = func_info.parameters
            tool = TOOL_DESC.format(
                name=name,
                description=description,
                parameters=parameters,
            )
            tools_text.append(tool)
            tools_name_text.append(name)
        tools_text = "\n\n".join(tools_text)
        tools_name_text = ", ".join(tools_name_text)
        tool_system += "\n\n" + TOOL_TEMPLATE.format(
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            tools_text=tools_text,
            tools_name_text=tools_name_text,
        )
        # 用户强制调用工具
        # if request.tool_choice is not None and messages[-1].role == "user":
        if request.tool_choice is not None:
            last_user_message = next(
                (m for m in reversed(messages) if m.role == "user"), None
            )
            if isinstance(request.tool_choice, str) and last_user_message:
                if request.tool_choice == "auto":
                    last_user_message.content += SPECIAL_PREFIX_TEMPLATE_TOOL.format(
                        tool_names=tools_name_text
                    )
            elif isinstance(request.tool_choice, ToolChoice) and last_user_message:
                last_user_message.content += (
                    SPECIAL_PREFIX_TEMPLATE_TOOL_FOR_CHAT.format(
                        tool_names=request.tool_choice.function.name
                    )
                )
            else:
                logger.error(
                    "Invalid request: tool_choices must be str or ToolChoices."
                )
        messages[0].content += tool_system.lstrip("\n").rstrip()
    # 将修改后的system prompt 作为第一条 加入
    result_messages = []
    # 将 工具调用的结果 组装到content
    for m_idx, m in enumerate(messages):
        # 将历史message 拼装到 prompt中
        if m.role in ("user", "system"):
            result_messages.append(m)
        elif m.role == "assistant":
            # 助手 消息 有工具调用 信息 回填
            as_content = m.content
            if as_content:
                # 有content，则 不填充 工具调用
                as_content = m.content.lstrip("\n").rstrip()
                result_messages.append(
                    ChatMessage(content=as_content, role="assistant")
                )
            else:
                as_content = ""
                if m.function_calls:
                    f_name, f_args = (
                        m.function_calls.name,
                        m.function_calls.arguments,
                    )
                else:
                    f_name, f_args = (
                        m.tool_calls[0].function.name,
                        m.tool_calls[0].function.arguments,
                    )
                as_content += f"Action:{f_name}\n Action Input:{f_args}"
                result_messages.append(
                    ChatMessage(content=as_content, role="assistant")
                )
        elif m.role in ("tool", "function"):
            # 工具调用结果信息回填 Observation: <result>工具返回的结果</result> 包括
            t_content = m.content.lstrip("\n").rstrip()
            tool_content = f"\nObservation: <result>{t_content}</result>\n"
            # assistant_message = result_messages[-1]
            # assistant_message.content += tool_content
            result_messages.append(ChatMessage(content=tool_content, role="assistant"))
        else:
            logger.warning("未知角色")
            result_messages.append(m)

    request.messages = result_messages
    return request


def parse_response(response, index):
    func_name, func_args = "", ""
    i = response.rfind(ACTION_TOKEN)
    j = response.rfind(ARGS_TOKEN)
    k = response.rfind(OBSERVATION_TOKEN)
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is omitted by the LLM,
            # because the output text may have discarded the stop word.
            response = response.rstrip() + OBSERVATION_TOKEN  # Add it back.
        k = response.rfind(OBSERVATION_TOKEN)
        func_name = response[i + len(ACTION_TOKEN) : j].strip()
        func_args = response[j + len(ARGS_TOKEN) : k].strip()
    if func_name:
        tool = ToolCallsMessage(
            id=str(uuid.uuid4()),
            type="function",
            function=FunctionCallsMessage(name=func_name, arguments=func_args),
        )
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatCompletionResponseMessage(
                role="assistant",
                tool_calls=[tool],
            ),
            finish_reason="tool_calls",
        )
        return choice_data
    # 检测到最终答案 不再调用工具
    z = response.rfind(ANSWER_TOKEN)
    if z >= 0:
        response = response[z + len(ANSWER_TOKEN) :]
    choice_data = ChatCompletionResponseChoice(
        index=index,
        message=ChatCompletionResponseMessage(content=response),
        finish_reason="stop",
    )
    return choice_data


async def get_gen_params(
    model_name: str,
    worker_addr: str,
    messages: Union[str, List[ChatMessage]],
    *,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    presence_penalty: Optional[float],
    frequency_penalty: Optional[float],
    max_tokens: Optional[int],
    echo: Optional[bool],
    stop: Optional[Union[str, List[str]]],
) -> Dict[str, Any]:
    conv = await get_conv(model_name, worker_addr)
    logger.debug(f"model conv: {conv}")
    conv = Conversation(
        name=conv["name"],
        system_template=conv["system_template"],
        system_message=conv["system_message"],
        roles=conv["roles"],
        messages=list(conv["messages"]),  # prevent in-place modification
        offset=conv["offset"],
        sep_style=SeparatorStyle(conv["sep_style"]),
        sep=conv["sep"],
        sep2=conv["sep2"],
        stop_str=conv["stop_str"],
        stop_token_ids=conv["stop_token_ids"],
    )

    if isinstance(messages, str):
        prompt = messages
    else:
        for message in messages:
            msg_role = message.role
            if msg_role == "system":
                conv.set_system_message(message.content)
            elif msg_role == "user":
                conv.append_message(conv.roles[0], message.content)
            elif msg_role == "assistant":
                conv.append_message(conv.roles[1], message.content)
            else:
                raise ValueError(f"Unknown role: {msg_role}")

        # Add a blank message for the assistant.
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "max_new_tokens": max_tokens,
        "echo": echo,
        "stop_token_ids": conv.stop_token_ids,
    }
    new_stop = set()
    _add_to_set(stop, new_stop)
    _add_to_set(conv.stop_str, new_stop)

    gen_params["stop"] = list(new_stop)

    return gen_params


@app.get("/v1/models", dependencies=[Depends(check_api_key)])
async def show_available_models_for_tool():
    return await show_available_models()


@app.post("/v1/chat/completions", dependencies=[Depends(check_api_key)])
async def create_chat_completion_for_tool(request: ChatCompletionRequest):
    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret
    logger.info(f"origin request: {request}")
    request = parse_function_messages(request)
    logger.info(f"parse request: {request}")
    worker_addr = await get_worker_address(request.model)

    gen_params = await get_gen_params(
        request.model,
        worker_addr,
        request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        max_tokens=request.max_tokens,
        echo=False,
        stop=request.stop,
    )
    logger.debug(f"gen_params: {gen_params}")
    max_new_tokens, error_check_ret = await check_length(
        request,
        gen_params["prompt"],
        gen_params["max_new_tokens"],
        worker_addr,
    )

    if error_check_ret is not None:
        return error_check_ret

    gen_params["max_new_tokens"] = max_new_tokens

    # todo  流式 后处理
    if request.stream:
        generator = chat_completion_stream_generator(
            request.model, gen_params, request.n, worker_addr
        )
        return StreamingResponse(generator, media_type="text/event-stream")

    choices = []
    chat_completions = []
    for i in range(request.n):
        content = asyncio.create_task(generate_completion(gen_params, worker_addr))
        chat_completions.append(content)
    try:
        all_tasks = await asyncio.gather(*chat_completions)
    except Exception as e:
        return create_error_response(ErrorCode.INTERNAL_ERROR, str(e))
    usage = UsageInfo()
    for i, content in enumerate(all_tasks):
        logger.info(f"llm response: {content}")
        if isinstance(content, str):
            content = json.loads(content)

        if content["error_code"] != 0:
            return create_error_response(content["error_code"], content["text"])
        if request.functions or request.tools:
            choices.append(parse_response(content["text"], i))
        else:
            choices.append(
                ChatCompletionResponseChoice(
                    index=i,
                    message=ChatCompletionResponseMessage(
                        content=content["text"], role="assistant"
                    ),
                    finish_reason=content.get("finish_reason", "stop"),
                )
            )
        logger.debug(f"choices: {choices}")
        if "usage" in content:
            task_usage = UsageInfo.model_validate(content["usage"])
            for usage_key, usage_value in task_usage.model_dump().items():
                setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    return ChatCompletionResponse(model=request.model, choices=choices, usage=usage)


@app.post("/v1/completions", dependencies=[Depends(check_api_key)])
async def create_completion_for_tool(request: CompletionRequest):
    return await create_completion(request)


@app.post("/v1/embeddings", dependencies=[Depends(check_api_key)])
@app.post("/v1/engines/{model_name}/embeddings", dependencies=[Depends(check_api_key)])
async def create_embeddings_for_tool(
    request: EmbeddingsRequest, model_name: str = None
):
    """Creates embeddings for the text"""
    return await create_embeddings(request, model_name)


@app.post("/api/v1/token_check")
async def count_tokens_for_tool(request: APITokenCheckRequest):
    """
    Checks the token count for each message in your list
    This is not part of the OpenAI API spec.
    """
    return await count_tokens(request)


@app.post("/api/v1/chat/completions")
async def create_chat_completion_for_tool(request: APIChatCompletionRequest):
    """Creates a completion for the chat message"""
    return await create_chat_completion(request)


if __name__ == "__main__":
    args = create_openai_api_server()
    if args.ssl:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            ssl_keyfile=os.environ["SSL_KEYFILE"],
            ssl_certfile=os.environ["SSL_CERTFILE"],
        )
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
