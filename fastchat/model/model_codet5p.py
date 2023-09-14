import gc
from threading import Thread
import torch
import transformers
from transformers import (
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)


@torch.inference_mode()
def generate_stream_codet5p(
    model,
    tokenizer,
    params,
    device,
    context_len=2048,
    stream_interval=2,
    judge_sent_end=False,
):
    prompt = params["prompt"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", 50))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 1024))
    stop_token_ids = params.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    decode_config = dict(skip_special_tokens=True, clean_up_tokenization_spaces=True)
    streamer = TextIteratorStreamer(tokenizer, **decode_config)
    encoding = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = encoding.input_ids
    encoding["decoder_input_ids"] = encoding["input_ids"].clone()
    input_echo_len = len(input_ids)

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=temperature >= 1e-5,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=10,
        top_p=top_p,
        top_k=top_k,
        eos_token_id=stop_token_ids,
    )

    class CodeBlockStopper(StoppingCriteria):
        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            # Code-completion is open-end generation.
            # We check \n\n to stop at end of a code block.
            if list(input_ids[0][-2:]) == [628, 198]:
                return True
            return False

    gen_kwargs = dict(
        **encoding,
        streamer=streamer,
        generation_config=generation_config,
        stopping_criteria=StoppingCriteriaList([CodeBlockStopper()]),
    )
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    i = 0
    output = ""
    for new_text in streamer:
        i += 1
        output += new_text
        if i % stream_interval == 0 or i == max_new_tokens - 1:
            yield {
                "text": output,
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": i,
                    "total_tokens": input_echo_len + i,
                },
                "finish_reason": None,
            }
        if i >= max_new_tokens:
            break

    if i >= max_new_tokens:
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
    thread.join()

    # clean
    gc.collect()
    torch.cuda.empty_cache()
    if device == "xpu":
        torch.xpu.empty_cache()
    if device == "npu":
        torch.npu.empty_cache()
