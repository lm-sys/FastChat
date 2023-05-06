import torch
from typing import List, Tuple

@torch.no_grad()
def stream_chat_token_num(tokenizer, query: str, history: List[Tuple[str, str]] = None):
    if history is None:
        history = []
    if not history:
        prompt = query
    else:
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
        prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
    inputs = tokenizer([prompt], return_tensors="pt")
    return torch.numel(inputs['input_ids'])

@torch.inference_mode()
def chatglm_generate_stream(
    model, tokenizer, params, device, context_len=2048, stream_interval=2
):
    """Generate text using model's chat api"""
    messages = params["prompt"]
    max_new_tokens = int(params.get("max_new_tokens", 256))
    temperature = float(params.get("temperature", 1.0))
    top_p = float(params.get("top_p", 0.7))
    echo = params.get("echo", True)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "top_p": top_p,
        "temperature": temperature,
        "logits_processor": None,
    }

    hist = []
    for i in range(0, len(messages) - 2, 2):
        hist.append((messages[i][1], messages[i + 1][1]))
    query = messages[-2][1]

    input_echo_len = stream_chat_token_num(tokenizer, query, hist)

    ret = None

    for i, (response, new_hist) in enumerate(model.stream_chat(tokenizer, query, hist)):
        if echo:
            output = query + " " + response
        else:
            output = response

        # Yield previous iteration output. Find the last iteration and set finish_reason value
        if ret is not None:
            yield ret

        ret = {
            "text": output,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": i,
                "total_tokens": input_echo_len + i,
            },
            "finish_reason": None
        }

    # TODO: ChatGLM stop when it reach max length
    # Here is last generation result, set finish_reason as stop
    ret = {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": "stop"
    }
    yield ret