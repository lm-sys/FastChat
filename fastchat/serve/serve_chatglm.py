import torch
from typing import List, Tuple


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

    for response, new_hist in model.stream_chat(tokenizer, query, hist):
        if echo:
            output = query + " " + response
        else:
            output = response

        yield output
