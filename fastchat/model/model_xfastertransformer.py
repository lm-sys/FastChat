import gc
from threading import Thread, Lock
import torch
from transformers import TextIteratorStreamer

lock = Lock()


@torch.inference_mode()
def generate_stream_xft(
    model,
    tokenizer,
    params,
    device,
    context_len=8192,
    stream_interval=2,
    judge_sent_end=False,
):
    prompt = params["prompt"]

    max_new_tokens = int(params.get("max_new_tokens", 4096))
    echo = params.get("echo", True)

    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    input_echo_len = len(inputs[0])
    max_len = max_new_tokens + input_echo_len

    decode_config = dict(skip_special_tokens=True, clean_up_tokenization_spaces=True)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, **decode_config)
    generation_kwargs = {
        "input_ids": inputs,
        "streamer": streamer,
        "max_length": max_len,
        "num_beams": model.config.beam_width,
    }
    if model.config.eos_token_id != -1:
        generation_kwargs["eos_token_id"] = model.config.eos_token_id
    if model.config.pad_token_id != -1:
        generation_kwargs["pad_token_id"] = model.config.pad_token_id
    if model.config.stop_words_ids is not None:
        generation_kwargs["stop_words_ids"] = model.config.stop_words_ids
    if model.config.do_sample:
        generation_kwargs["do_sample"] = True
    if model.config.temperature > 0:
        generation_kwargs["temperature"] = model.config.temperature
    if model.config.top_p > 0:
        generation_kwargs["top_p"] = model.config.top_p
    if model.config.top_k > 0:
        generation_kwargs["top_k"] = model.config.top_k
    if model.config.repetition_penalty > 0:
        generation_kwargs["repetition_penalty"] = model.config.repetition_penalty
    if model.config.early_stopping:
        generation_kwargs["early_stopping"] = model.config.early_stopping
    thread = Thread(target=model.model.generate, kwargs=generation_kwargs)
    lock.acquire()
    thread.start()
    if echo:
        output = prompt
    else:
        output = ""
    i = 0
    for i, new_text in enumerate(streamer):
        output += new_text
        yield {
            "text": output,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": i,
                "total_tokens": input_echo_len + i,
            },
            "finish_reason": None,
        }
    output = output.strip()
    if i == max_new_tokens - 1:
        finish_reason = "length"
    else:
        finish_reason = "stop"
    yield {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }
    lock.release()
    thread.join()
    gc.collect()
