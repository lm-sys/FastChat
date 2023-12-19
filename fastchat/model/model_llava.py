"""
Inference code for Llava.
Adapted from https://huggingface.co/spaces/badayvedat/LLaVA/blob/main/llava/serve/model_worker.py and
https://github.com/haotian-liu/LLaVA/blob/5da97161b9e2c3ae19b1d4a39eeb43148091d728/llava/mm_utils.py
"""

from io import BytesIO
import base64
import os
import requests
from threading import Thread

import torch
from transformers import TextIteratorStreamer

from fastchat.utils import load_image

@torch.inference_mode()
def generate_stream_llava(
    model,
    processor,
    params,
    device,
    context_len=2048,
    stream_interval=2,
    judge_sent_end=False,
):
    prompt = params["prompt"]
    images = params.get("images", None)
    temperature = float(params.get("temperature", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_context_length = getattr(model.config, "max_position_embeddings", 2048)
    max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
    stop_str = params.get("stop", None)
    do_sample = True if temperature > 0.001 else False
    echo = params.get("echo", False)

    ori_prompt = prompt
    if type(stop_str) is list:
        stop_str = stop_str[0]

    do_sample = True if temperature > 0.001 else False

    streamer = TextIteratorStreamer(
        processor.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15
    )

    if images is not None and len(images) > 0:
        images = [load_image(image) for image in images]
        inputs = processor(prompt, images, return_tensors='pt').to(device)
    else:
        inputs = processor.tokenizer(prompt, return_tensors='pt').to(device)

    max_new_tokens = min(
        max_new_tokens, max_context_length - inputs["input_ids"].shape[-1]
    )

    if max_new_tokens < 1:
        yield {
            "text": ori_prompt
            + "Exceeds max token length. Please start a new conversation, thanks.",
            "error_code": 0,
        }
        return

    thread = Thread(
        target=model.generate,
        kwargs=dict(
            **inputs,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
        ),
    )
    thread.start()

    if echo:
        generated_text = ori_prompt
    else:
        generated_text = ""

    generated_tokens = 0
    finish_reason = None
    for new_text in streamer:
        generated_text += new_text
        generated_tokens += len(processor.tokenizer.encode(new_text))
        if generated_text.endswith(stop_str):
            finish_reason = "stop"
            break
        elif generated_tokens >= max_new_tokens:
            finish_reason = "length"
            break

        yield {
            "text": generated_text,
            "usage": {
                "prompt_tokens": inputs["input_ids"].shape[-1],
                "completion_tokens": generated_tokens,
                "total_tokens": inputs["input_ids"].shape[-1] + generated_tokens,
            },
            "finish_reason": None,
        }

    yield {
        "text": generated_text,
        "usage": {
            "prompt_tokens": inputs["input_ids"].shape[-1],
            "completion_tokens": generated_tokens,
            "total_tokens": inputs["input_ids"].shape[-1] + generated_tokens,
        },
        "finish_reason": finish_reason,
    }
