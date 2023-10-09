import gc
import sys
from typing import Dict

import torch


def generate_stream_exllama(
    model,
    tokenizer,
    params: Dict,
    device: str,
    context_len: int,
    stream_interval: int = 2,
    judge_sent_end: bool = False,
):
    try:
        from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2Sampler
    except ImportError as e:
        print(f"Error: Failed to load Exllamav2. {e}")
        sys.exit(-1)

    prompt = params["prompt"]

    generator = ExLlamaV2StreamingGenerator(model.model, model.cache, tokenizer)
    settings = ExLlamaV2Sampler.Settings()

    settings.temperature = float(params.get("temperature", 0.85))
    settings.top_k = int(params.get("top_k", 50))
    settings.top_p = float(params.get("top_p", 0.8))
    settings.token_repetition_penalty = float(params.get("repetition_penalty", 1.15))
    settings.disallow_tokens(generator.tokenizer, [generator.tokenizer.eos_token_id])

    max_new_tokens = int(params.get("max_new_tokens", 256))

    generator.set_stop_conditions(params.get("stop_token_ids", None) or [])
    echo = bool(params.get("echo", True))

    input_ids = generator.tokenizer.encode(prompt)
    prompt_tokens = input_ids.shape[-1]
    generator.begin_stream(input_ids, settings)

    generated_tokens = 0
    if echo:
        output = prompt
    else:
        output = ""
    while True:
        chunk, eos, _ = generator.stream()
        output += chunk
        generated_tokens += 1
        if generated_tokens == max_new_tokens:
            finish_reason = "length"
            break
        elif eos:
            finish_reason = "length"
            break
        yield {
            "text": output,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": generated_tokens,
                "total_tokens": prompt_tokens + generated_tokens,
            },
            "finish_reason": None,
        }

    yield {
        "text": output,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": generated_tokens,
            "total_tokens": prompt_tokens + generated_tokens,
        },
        "finish_reason": finish_reason,
    }
    gc.collect()
