import torch
import gc

import os
import time
import random
from typing import Dict, Optional, Sequence, List, Tuple
from transformers.cache_utils import Cache, DynamicCache
from transformers import (
    LlamaModel,
    LlamaForCausalLM,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
import torch.nn.functional as F


def get_jacobian_trajectory(
    model, tokenizer, input_ids, attention_mask, max_new_tokens
):
    bsz = input_ids.shape[0]
    prompt_len = [torch.sum(t) for t in attention_mask]
    max_prompt_len = max(prompt_len)
    total_len = max_prompt_len + max_new_tokens

    # initialize the first point of jacobian trajectory
    tokens = torch.full(
        (bsz, total_len), tokenizer.pad_token_id, dtype=torch.long, device=model.device
    )
    for i in range(bsz):
        tokens[i, :] = torch.tensor(
            random.choices(input_ids[i][attention_mask[i] == 1], k=total_len),
            dtype=torch.long,
            device=model.device,
        )
        tokens[i, : prompt_len[i]] = input_ids[i][: prompt_len[i]].to(
            dtype=torch.long, device=model.device
        )
    itr = 0
    next_generation = tokens
    generate_attention_mask = torch.full_like(next_generation, 1).to(model.device)
    accurate_lengths = torch.tensor([prompt_len[i].item()] * bsz, device=model.device)
    prev_len = 0
    while True:
        current_generation = next_generation
        with torch.no_grad():
            logits = model(current_generation, generate_attention_mask).logits
        next_generation = torch.argmax(
            torch.nn.functional.softmax(logits, dim=-1) / 0.001, dim=-1
        )

        # hold prompt unchanged and update generated tokens
        for i in range(bsz):
            next_generation[i, :] = torch.cat(
                (
                    tokens[i, : prompt_len[i]],
                    next_generation[i, prompt_len[i] - 1 : total_len - 1],
                ),
                dim=0,
            )

        if (
            torch.all(torch.eq(next_generation, current_generation)).item()
            and itr == max_new_tokens
            or len(
                torch.where(
                    current_generation[0, : accurate_lengths[0]]
                    == tokenizer.eos_token_id
                )[0]
            )
            > 0
        ):
            # forced exit due to max_new_tokens constraint or eos reached
            return next_generation, itr

        # skip the first itr, current_generation has not been updated yet
        if itr != 0:
            if torch.all(torch.eq(next_generation, current_generation)).item():
                matched_position = total_len
            else:
                matched_position = (
                    torch.eq(current_generation, next_generation).squeeze(0) == False
                ).nonzero(as_tuple=True)[0][0]
            fast_forward_cnt = matched_position - accurate_lengths[0]

            for i in range(bsz):
                accurate_lengths[i] = matched_position.item()

            # flush and print the first sequence
            generated_str = tokenizer.decode(
                next_generation[0, prompt_len[0] : accurate_lengths[0]],
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            print(generated_str[prev_len:], flush=True, end="")
            prev_len = len(generated_str)

            if torch.all(torch.eq(next_generation, current_generation)).item():
                # early termination: itr < max_new_tokens
                return next_generation, itr

        itr += 1


def generate_stream_cllm(
    model,
    tokenizer,
    params,
    device,
    context_len,
    stream_interval=2,
    judge_sent_end=False,
):
    # converge_step = []
    prompt = params["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    max_new_tokens = int(params.get("n_token_seq_length", 32))
    max_new_seq_len = int(params.get("max_new_tokens", 1024))

    prompt_len = torch.sum(inputs["attention_mask"], dim=-1)
    generation = inputs["input_ids"]
    input_echo_len = len(generation)

    ### generation phase
    itr = 0
    eos_reached = False
    while True:
        if itr == 0:
            input_ids = inputs["input_ids"]
            input_masks = inputs["attention_mask"]
        else:
            input_masks = torch.ones_like(input_ids).to(device)
            for j in range(bsz):
                input_masks[j][
                    torch.sum(inputs["attention_mask"], dim=-1)[j]
                    + itr * max_new_tokens :
                ] = 0

        bsz = input_ids.shape[0]
        eos_reached = torch.tensor([False] * bsz, device=device)

        generation, iter_steps = get_jacobian_trajectory(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=input_masks,
            max_new_tokens=max_new_tokens,
        )

        ### inspect <eos>
        for j in range(bsz):
            prompt_len = torch.sum(input_masks, dim=-1)
            eos_positions = torch.where(generation[j] == tokenizer.eos_token_id)[0]

            if len(eos_positions) == 0:
                # no EOS, continue to the next item in the batch
                generation[j][prompt_len[j] + max_new_tokens :] = tokenizer.pad_token_id
                continue
            # otherwise, set tokens coming after EOS as pad
            else:
                if len(eos_positions) != 0:
                    eos_reached[j] = True
                    generation[j, int(eos_positions[0]) + 1 :] = tokenizer.pad_token_id

        itr += 1

        if all(eos_reached) or itr * max_new_tokens >= max_new_seq_len:
            break
        input_ids = generation[
            torch.where(eos_reached == False)[0].tolist(), ...
        ]  # delete samples with <eos> generated

    if all(eos_reached):
        finish_reason = "eos"
    elif itr * max_new_tokens > max_new_seq_len:
        finish_reason = "length"
    else:
        finish_reason = "stop"

    output = tokenizer.decode(input_ids[0], skip_special_tokens=False)

    yield {
        "text": "",
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": itr * max_new_tokens,
            "total_tokens": input_echo_len + itr * max_new_tokens,
        },
        "finish_reason": finish_reason,
    }

    # clean
    gc.collect()
    torch.cuda.empty_cache()
    if device == "xpu":
        torch.xpu.empty_cache()
    if device == "npu":
        torch.npu.empty_cache()
