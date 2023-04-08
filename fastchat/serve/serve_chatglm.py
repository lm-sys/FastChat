import torch
from typing import List, Tuple

def chatglm_generate_stream(tokenizer, model, params, device,
                    context_len=2048, stream_interval=2):
    """Generate text using model's chat api"""

    query = params["prompt"]
    max_length = int(params.get("max_new_tokens", 256))
    temperature = float(params.get("temperature", 1.0))
    top_p = float(params.get("top_p", 0.7))

    gen_kwargs = {
        "max_length": max_length,
        "do_sample": True,
        "top_p": top_p,
        "temperature": temperature,
        "logits_processor": None
    }


    history = params["conv"].messages

    prompt = ""
    i = 0
    while(i < len(history)-2):
        prompt += "[Round {}]\n".format(i)
        prompt += history[i][0] + "\n"
        prompt += history[i][1] +"\n"
        i += 1
    prompt += "[Round {}]\n{}\n".format(i, history[i][1])

    hist = []
    for token , hist in model.stream_chat(tokenizer,prompt,hist):
        output = query + "#" + token + " "
        yield output