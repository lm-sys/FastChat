import os
import gc
from threading import Thread
from types import SimpleNamespace
from threading import Thread
from transformers import TextIteratorStreamer
import warnings

import torch


class MambaModel:
    def __init__(self, tokenizer, model):
        self.model = model
        self.tokenizer = tokenizer
        self.config = model.config

    def to(self, target):
        assert target == "cuda"
        self.model.to(target)

    def generate(self, input_ids, do_sample, temperature, max_new_tokens):
        generation_kwargs = dict(
            input_ids=input_ids, max_length=max_new_tokens, eos_token_id=200
        )
        output = self.model.generate(**generation_kwargs)
        return output


@torch.inference_mode()
def generate_stream_mamba(
    model,
    tokenizer,
    params,
    device: str,
    context_len: int,
    stream_interval: int = 2,
    judge_sent_end: bool = False,
):
    ctx = params["prompt"].strip()
    max_new_tokens = int(params.get("max_new_tokens", 2048))
    streamer = TextIteratorStreamer(tokenizer)
    inputs = tokenizer(ctx, return_tensors="pt").to(device)
    inputs.pop("token_type_ids", None)
    inputs.pop("attention_mask", None)
    generation_kwargs = dict(**inputs, max_length=max_new_tokens, streamer=streamer)
    thread = Thread(target=model.model.generate, kwargs=generation_kwargs)
    thread.start()
    output_str = ""
    seqlen_og = inputs["input_ids"].shape[1]
    # __import__("ipdb").set_trace()
    # for _ in range(seqlen_og):
    #     item = next(streamer)
    repeating = True
    for new_text in streamer:
        if new_text.strip() in ctx and repeating:
            # get rid of the repeated input ids
            repeating = True
            continue
        else:
            repeating = False
        output_str += new_text
        if "\x17" in new_text:
            break
        yield {"text": output_str}
    yield {"text": output_str}
