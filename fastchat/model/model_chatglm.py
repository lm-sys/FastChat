"""
Inference code for ChatGLM.
Adapted from https://huggingface.co/THUDM/chatglm-6b/blob/main/modeling_chatglm.py.
"""
import re

import torch
from transformers.generation.logits_process import LogitsProcessor


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


invalid_score_processor = InvalidScoreLogitsProcessor()


def process_response(response):
    response = response.strip()
    response = response.replace("[[训练时间]]", "2023年")
    punkts = [
        [",", "，"],
        ["!", "！"],
        [":", "："],
        [";", "；"],
        ["\?", "？"],
    ]
    for item in punkts:
        response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
        response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
    return response


def recover_message_list(prompt):
    role_token_pattern = "|".join(
        [re.escape(r) for r in ["<|system|>", "<|user|>", "<|assistant|>"]]
    )
    role = None
    last_end_idx = -1
    message_list = []
    for match in re.finditer(role_token_pattern, prompt):
        if role:
            messge = {}
            if role == "<|system|>":
                messge["role"] = "system"
            elif role == "<|user|>":
                messge["role"] = "user"
            else:
                messge["role"] = "assistant"
            messge["content"] = prompt[last_end_idx + 1 : match.start()]
            message_list.append(messge)

        role = prompt[match.start() : match.end()]
        last_end_idx = match.end()

    return message_list


@torch.inference_mode()
def generate_stream_chatglm(
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
    max_new_tokens = int(params.get("max_new_tokens", 256))
    echo = params.get("echo", True)

    model_type = str(type(model)).lower()
    if "peft" in model_type:
        model_type = str(type(model.base_model.model)).lower()

    if "chatglm3" in model_type:
        message_list = recover_message_list(prompt)
        inputs = tokenizer.build_chat_input(
            query=message_list[-1]["content"], history=message_list[:-1], role="user"
        ).to(model.device)
    else:
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    input_echo_len = len(inputs["input_ids"][0])

    gen_kwargs = {
        "max_length": max_new_tokens + input_echo_len,
        "do_sample": True if temperature > 1e-5 else False,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "logits_processor": [invalid_score_processor],
    }
    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature

    total_len = 0
    for total_ids in model.stream_generate(**inputs, **gen_kwargs):
        total_ids = total_ids.tolist()[0]
        total_len = len(total_ids)
        if echo:
            output_ids = total_ids
        else:
            output_ids = total_ids[input_echo_len:]
        response = tokenizer.decode(output_ids)
        response = process_response(response)

        yield {
            "text": response,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": total_len - input_echo_len,
                "total_tokens": total_len,
            },
            "finish_reason": None,
        }

    # TODO: ChatGLM stop when it reach max length
    # Only last stream result contains finish_reason, we set finish_reason as stop
    ret = {
        "text": response,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": total_len - input_echo_len,
            "total_tokens": total_len,
        },
        "finish_reason": "stop",
    }
    yield ret
