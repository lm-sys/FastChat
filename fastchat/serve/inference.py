"""Inference for FastChat models."""
import abc
import gc
import json
import math
import os
import sys
import time
from typing import Iterable, Optional, Dict
import warnings

import psutil
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModel,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    AutoConfig,
)
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from fastchat.conversation import get_conv_template, SeparatorStyle
from fastchat.model.model_adapter import (
    load_model,
    get_conversation_template,
    get_generate_stream_function,
)
from fastchat.modules.awq import AWQConfig
from fastchat.modules.gptq import GptqConfig
from fastchat.modules.exllama import ExllamaConfig
from fastchat.modules.xfastertransformer import XftConfig
from fastchat.utils import is_partial_stop, is_sentence_complete, get_context_length


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


@torch.inference_mode()
def generate_stream(
    model,
    tokenizer,
    params: Dict,
    device: str,
    context_len: int,
    stream_interval: int = 2,
    judge_sent_end: bool = False,
):
    if hasattr(model, "device"):
        device = model.device

    # Read parameters
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    logprobs = params.get("logprobs", None)  # FIXME: Support logprobs>1.
    echo = bool(params.get("echo", True))
    stop_str = params.get("stop", None)
    stop_token_ids = params.get("stop_token_ids", None) or []
    if tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )
    input_ids = tokenizer(prompt).input_ids

    if model.config.is_encoder_decoder:
        max_src_len = context_len
    else:  # truncate
        max_src_len = context_len - max_new_tokens - 1

    input_ids = input_ids[-max_src_len:]
    output_ids = list(input_ids)
    input_echo_len = len(input_ids)

    if model.config.is_encoder_decoder:
        if logprobs is not None:  # FIXME: Support logprobs for encoder-decoder models.
            raise NotImplementedError
        encoder_output = model.encoder(
            input_ids=torch.as_tensor([input_ids], device=device)
        )[0]
        start_ids = torch.as_tensor(
            [[model.generation_config.decoder_start_token_id]],
            dtype=torch.int64,
            device=device,
        )
    else:
        start_ids = torch.as_tensor([input_ids], device=device)

    past_key_values = out = None
    token_logprobs = [None]  # The first token has no logprobs.
    sent_interrupt = False
    finish_reason = None
    stopped = False
    for i in range(max_new_tokens):
        if i == 0:  # prefill
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=start_ids,
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                )
                logits = model.lm_head(out[0])
            else:
                out = model(input_ids=start_ids, use_cache=True)
                logits = out.logits
            past_key_values = out.past_key_values

            if logprobs is not None:
                # Prefull logprobs for the prompt.
                shift_input_ids = start_ids[..., 1:].contiguous()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_logits = torch.log_softmax(shift_logits, dim=-1).tolist()
                for label_id, logit in zip(
                    shift_input_ids[0].tolist(), shift_logits[0]
                ):
                    token_logprobs.append(logit[label_id])
        else:  # decoding
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=torch.as_tensor(
                        [[token] if not sent_interrupt else output_ids],
                        device=device,
                    ),
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                    past_key_values=past_key_values if not sent_interrupt else None,
                )
                sent_interrupt = False

                logits = model.lm_head(out[0])
            else:
                out = model(
                    input_ids=torch.as_tensor(
                        [[token] if not sent_interrupt else output_ids],
                        device=device,
                    ),
                    use_cache=True,
                    past_key_values=past_key_values if not sent_interrupt else None,
                )
                sent_interrupt = False
                logits = out.logits
            past_key_values = out.past_key_values

        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
            tokens = [int(token) for token in indices.tolist()]
        token = tokens[0]
        output_ids.append(token)
        if logprobs is not None:
            # Cannot use last_token_logits because logprobs is based on raw logits.
            token_logprobs.append(
                torch.log_softmax(logits[0, -1, :], dim=-1)[token].tolist()
            )

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        # Yield the output tokens
        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            ret_logprobs = None
            if logprobs is not None:
                ret_logprobs = {
                    "text_offset": [],
                    "tokens": [
                        tokenizer.decode(token)
                        for token in (
                            output_ids if echo else output_ids[input_echo_len:]
                        )
                    ],
                    "token_logprobs": token_logprobs
                    if echo
                    else token_logprobs[input_echo_len:],
                    "top_logprobs": [{}]
                    * len(token_logprobs if echo else token_logprobs[input_echo_len:]),
                }
                # Compute text_offset
                curr_pos = 0
                for text in ret_logprobs["tokens"]:
                    ret_logprobs["text_offset"].append(curr_pos)
                    curr_pos += len(text)

            # TODO: For the issue of incomplete sentences interrupting output, apply a patch and others can also modify it to a more elegant way
            if judge_sent_end and stopped and not is_sentence_complete(output):
                if len(tokens) > 1:
                    token = tokens[1]
                    output_ids[-1] = token
                else:
                    output_ids.pop()
                stopped = False
                sent_interrupt = True

            partially_stopped = False
            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                    else:
                        partially_stopped = is_partial_stop(output, stop_str)
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            break
                        else:
                            partially_stopped = is_partial_stop(output, each_stop)
                            if partially_stopped:
                                break
                else:
                    raise ValueError("Invalid stop field type.")

            # Prevent yielding partial stop sequence
            if not partially_stopped:
                yield {
                    "text": output,
                    "logprobs": ret_logprobs,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": i,
                        "total_tokens": input_echo_len + i,
                    },
                    "finish_reason": None,
                }

        if stopped:
            break

    # Finish stream event, which contains finish reason
    else:
        finish_reason = "length"

    if stopped:
        finish_reason = "stop"

    yield {
        "text": output,
        "logprobs": ret_logprobs,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # Clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()
    if device == "xpu":
        torch.xpu.empty_cache()
    if device == "npu":
        torch.npu.empty_cache()


class ChatIO(abc.ABC):
    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str):
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream):
        """Stream output."""

    @abc.abstractmethod
    def print_output(self, text: str):
        """Print output."""


def chat_loop(
    model_path: str,
    device: str,
    num_gpus: int,
    max_gpu_memory: str,
    dtype: Optional[torch.dtype],
    load_8bit: bool,
    cpu_offloading: bool,
    conv_template: Optional[str],
    conv_system_msg: Optional[str],
    temperature: float,
    repetition_penalty: float,
    max_new_tokens: int,
    chatio: ChatIO,
    gptq_config: Optional[GptqConfig] = None,
    awq_config: Optional[AWQConfig] = None,
    exllama_config: Optional[ExllamaConfig] = None,
    xft_config: Optional[XftConfig] = None,
    revision: str = "main",
    judge_sent_end: bool = True,
    debug: bool = True,
    history: bool = True,
):
    # Model
    model, tokenizer = load_model(
        model_path,
        device=device,
        num_gpus=num_gpus,
        max_gpu_memory=max_gpu_memory,
        dtype=dtype,
        load_8bit=load_8bit,
        cpu_offloading=cpu_offloading,
        gptq_config=gptq_config,
        awq_config=awq_config,
        exllama_config=exllama_config,
        xft_config=xft_config,
        revision=revision,
        debug=debug,
    )
    generate_stream_func = get_generate_stream_function(model, model_path)

    model_type = str(type(model)).lower()
    is_t5 = "t5" in model_type
    is_codet5p = "codet5p" in model_type
    is_xft = "xft" in model_type

    # Hardcode T5's default repetition penalty to be 1.2
    if is_t5 and repetition_penalty == 1.0:
        repetition_penalty = 1.2

    # Set context length
    context_len = get_context_length(model.config)

    # Chat
    def new_chat():
        if conv_template:
            conv = get_conv_template(conv_template)
        else:
            conv = get_conversation_template(model_path)
        if conv_system_msg is not None:
            conv.set_system_message(conv_system_msg)
        return conv

    def reload_conv(conv):
        """
        Reprints the conversation from the start.
        """
        for message in conv.messages[conv.offset :]:
            chatio.prompt_for_output(message[0])
            chatio.print_output(message[1])

    conv = None

    while True:
        if not history or not conv:
            conv = new_chat()

        try:
            inp = chatio.prompt_for_input(conv.roles[0])
        except EOFError:
            inp = ""

        if inp == "!!exit" or not inp:
            print("exit...")
            break
        elif inp == "!!reset":
            print("resetting...")
            conv = new_chat()
            continue
        elif inp == "!!remove":
            print("removing last message...")
            if len(conv.messages) > conv.offset:
                # Assistant
                if conv.messages[-1][0] == conv.roles[1]:
                    conv.messages.pop()
                # User
                if conv.messages[-1][0] == conv.roles[0]:
                    conv.messages.pop()
                reload_conv(conv)
            else:
                print("No messages to remove.")
            continue
        elif inp == "!!regen":
            print("regenerating last message...")
            if len(conv.messages) > conv.offset:
                # Assistant
                if conv.messages[-1][0] == conv.roles[1]:
                    conv.messages.pop()
                # User
                if conv.messages[-1][0] == conv.roles[0]:
                    reload_conv(conv)
                    # Set inp to previous message
                    inp = conv.messages.pop()[1]
                else:
                    # Shouldn't happen in normal circumstances
                    print("No user message to regenerate from.")
                    continue
            else:
                print("No messages to regenerate.")
                continue
        elif inp.startswith("!!save"):
            args = inp.split(" ", 1)

            if len(args) != 2:
                print("usage: !!save <filename>")
                continue
            else:
                filename = args[1]

            # Add .json if extension not present
            if not "." in filename:
                filename += ".json"

            print("saving...", filename)
            with open(filename, "w") as outfile:
                json.dump(conv.dict(), outfile)
            continue
        elif inp.startswith("!!load"):
            args = inp.split(" ", 1)

            if len(args) != 2:
                print("usage: !!load <filename>")
                continue
            else:
                filename = args[1]

            # Check if file exists and add .json if needed
            if not os.path.exists(filename):
                if (not filename.endswith(".json")) and os.path.exists(
                    filename + ".json"
                ):
                    filename += ".json"
                else:
                    print("file not found:", filename)
                    continue

            print("loading...", filename)
            with open(filename, "r") as infile:
                new_conv = json.load(infile)

            conv = get_conv_template(new_conv["template_name"])
            conv.set_system_message(new_conv["system_message"])
            conv.messages = new_conv["messages"]
            reload_conv(conv)
            continue

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if is_codet5p:  # codet5p is a code completion model.
            prompt = inp

        gen_params = {
            "model": model_path,
            "prompt": prompt,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }

        try:
            chatio.prompt_for_output(conv.roles[1])
            output_stream = generate_stream_func(
                model,
                tokenizer,
                gen_params,
                device,
                context_len=context_len,
                judge_sent_end=judge_sent_end,
            )
            t = time.time()
            outputs = chatio.stream_output(output_stream)
            duration = time.time() - t
            conv.update_last_message(outputs.strip())

            if debug:
                num_tokens = len(tokenizer.encode(outputs))
                msg = {
                    "conv_template": conv.name,
                    "prompt": prompt,
                    "outputs": outputs,
                    "speed (token/s)": round(num_tokens / duration, 2),
                }
                print(f"\n{msg}\n")

        except KeyboardInterrupt:
            print("stopped generation.")
            # If generation didn't finish
            if conv.messages[-1][1] is None:
                conv.messages.pop()
                # Remove last user message, so there isn't a double up
                if conv.messages[-1][0] == conv.roles[0]:
                    conv.messages.pop()

                reload_conv(conv)
