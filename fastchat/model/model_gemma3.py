from threading import Thread
import gc
import torch
from transformers import TextIteratorStreamer

def generate_stream_gemma3(
    model,
    tokenizer,
    params,
    device,
    context_len,
    stream_interval=2,
    judge_sent_end=False
):
    """Custom generate stream function for Gemma-3 models"""
    # Get parameters from the request
    prompt = params.get("prompt", "")
    messages = params.get("messages", None)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    echo = bool(params.get("echo", True))
    stop_str = params.get("stop", None)
    stop_token_ids = params.get("stop_token_ids", None) or []
    model_name = params.get("model", None)

    if tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)

    is_base_model = "pt" in model_name.lower() or "base" in model_name.lower()

    if not is_base_model:
        # Format input based on whether we have messages or a plain prompt
        if messages:
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
            ).to(model.device)
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
            ).to(model.device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    input_ids = inputs["input_ids"]
    input_echo_len = input_ids.shape[1]

    # Configure generation parameters
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "temperature": temperature if temperature > 0.0 else 1.0,
    }

    if top_p < 1.0:
        generate_kwargs["top_p"] = top_p
    if top_k > 0:
        generate_kwargs["top_k"] = top_k
    if repetition_penalty > 1.0:
        generate_kwargs["repetition_penalty"] = repetition_penalty

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=not echo, skip_special_tokens=True)
    generate_kwargs["streamer"] = streamer

    # Start generation in a separate thread
    thread = Thread(target=lambda: model.generate(input_ids=input_ids, **generate_kwargs))
    thread.start()

    # Track generation progress
    generated_tokens = 0
    output_text = ""

    # Stream tokens
    for new_text in streamer:
        output_text += new_text
        generated_tokens += 1

        # Check for stop strings
        should_stop = False
        if stop_str:
            if isinstance(stop_str, str):
                if stop_str in output_text:
                    output_text = output_text[: output_text.find(stop_str)]
                    should_stop = True
            elif isinstance(stop_str, list):
                for stop in stop_str:
                    if stop in output_text:
                        output_text = output_text[: output_text.find(stop)]
                        should_stop = True
                        break

        # Stream at intervals or when stopping
        if generated_tokens % stream_interval == 0 or should_stop:
            yield {
                "text": output_text,
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": generated_tokens,
                    "total_tokens": input_echo_len + generated_tokens,
                },
                "finish_reason": "stop" if should_stop else None,
            }

        if should_stop:
            break

    # Final output with finish reason
    if thread.is_alive():
        thread.join(
            timeout=3600
        )  # Arbitrary value, but if it doesn't complete in this much time then something is wrong

    yield {
        "text": output_text,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": generated_tokens,
            "total_tokens": input_echo_len + generated_tokens,
        },
        "finish_reason": "length",
    }

    # Clean up
    gc.collect()
    torch.cuda.empty_cache()
    if device == "xpu":
        torch.xpu.empty_cache()
    if device == "npu":
        torch.npu.empty_cache()