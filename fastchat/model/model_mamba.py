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
        streamer = TextIteratorStreamer(self.tokenizer)
        generation_kwargs = dict(input_ids=input_ids, max_length=max_new_tokens, streamer=streamer)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        output_str = ""
        seqlen_og = input_ids.shape[1]
        for _ in range(seqlen_og):
            next(streamer)
        for new_text in streamer:
            output_str += new_text
            if '\x17' in new_text:
                break
        return [input_ids[0].tolist() + self.tokenizer.encode(output_str)]


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
    inputs.pop('token_type_ids', None)
    inputs.pop('attention_mask', None)
    generation_kwargs = dict(**inputs, max_length=max_new_tokens, streamer=streamer)
    thread = Thread(target=model.model.generate, kwargs=generation_kwargs)
    thread.start()
    output_str = ""
    seqlen_og = inputs['input_ids'].shape[1]
    for _ in range(seqlen_og):
        next(streamer)
    for new_text in streamer:
        output_str += new_text
        if '\x17' in new_text:
            break
        yield {'text': output_str}
    yield {'text': output_str}