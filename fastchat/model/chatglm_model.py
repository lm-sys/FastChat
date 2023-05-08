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
    top_p = float(params.get("top_p", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    echo = params.get("echo", True)

    gen_kwargs = {
        #"max_new_tokens": max_new_tokens,  disabled due to a warning.
        "do_sample": True if temperature > 1e-5 else False,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "logits_processor": None,
    }
    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature

    hist = []
    for i in range(0, len(messages) - 2, 2):
        hist.append((messages[i][1], messages[i + 1][1]))
    query = messages[-2][1]

    for response, new_hist in model.stream_chat(tokenizer, query, hist, **gen_kwargs):
        if echo:
            output = query + " " + response
        else:
            output = response

        yield output
