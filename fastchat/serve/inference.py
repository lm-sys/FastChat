"""Inference for FastChat models."""
import abc
import gc
import math

import sys
import time
from typing import Iterable, Optional, Dict
import warnings
import re

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
from fastchat.modules.gptq import GptqConfig
from fastchat.utils import is_partial_stop, is_sentence_complete, get_context_length

from api_call import extract_features


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
    model, tokenizer, params, device, context_len=2048, stream_interval=2
):
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    echo = bool(params.get("echo", True))
    stop_str = params.get("stop", None)
    stop_token_ids = params.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )

    # Print prompt
    # print(f"PROMPT: {prompt}\n")

    input_ids = tokenizer(prompt).input_ids

    if model.config.is_encoder_decoder:
        max_src_len = context_len
    else:  # truncate
        max_src_len = context_len - max_new_tokens - 1

    input_ids = input_ids[-max_src_len:]
    output_ids = list(input_ids)
    input_echo_len = len(input_ids)

    if model.config.is_encoder_decoder:
        encoder_output = model.encoder(
            input_ids=torch.as_tensor([input_ids], device=device)
        )[0]
        start_ids = torch.as_tensor(
            [[model.generation_config.decoder_start_token_id]],
            dtype=torch.int64,
            device=device,
        )

    past_key_values = out = None
    for i in range(max_new_tokens):
        if i == 0:
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=start_ids,
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                )
                logits = model.lm_head(out[0])
            else:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
            past_key_values = out.past_key_values
        else:
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=torch.as_tensor([[token]], device=device),
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                    past_key_values=past_key_values,
                )

                logits = model.lm_head(out[0])
            else:
                out = model(
                    input_ids=torch.as_tensor([[token]], device=device),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
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
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

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
            )
            partially_stopped = False
            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                        print(f"STOP HERE!!!")
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

            # prevent yielding partial stop sequence
            if not partially_stopped:
                yield {
                    "text": output,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": i,
                        "total_tokens": input_echo_len + i,
                    },
                    "finish_reason": None,
                }

        if stopped:
            break

    # finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif stopped:
        finish_reason = "stop"
    else:
        finish_reason = None

    # Print stream generated from inference
    """
    print(f"GENERATE STREAM: {output}")
    print("-----------" * 50)
    """

    yield {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()
    if device == "xpu":
        torch.xpu.empty_cache()


# Function to add API calling for inference
def generate_special_stream(
    model, tokenizer, gen_params, device, context_len=2048, stream_interval=2
):
    prompt = gen_params["prompt"]
    output_stream = generate_stream(model, tokenizer, gen_params, device)
    output_stream_list = list(output_stream)
    outputs = ""
    for data in output_stream_list:
        if not data:
            continue
        outputs = data["text"]

    if "Function:" in outputs and "Observation:" in outputs:
        new_outputs = extract_features(outputs)
        # Use regex to fix prompt
        prompt = re.search(r".*?(?=ASSISTANT:)", prompt).group(0)
        # Make sure the prompt has the same format as the prompt in dataset
        prompt = (
            prompt
            + "\nInput:\n"
            + outputs
            + new_outputs
            + "\nFinal Answer:  ASSISTANT: "
        )
        gen_params["prompt"] = prompt
        # Get the new inference result
        output_stream = generate_stream(model, tokenizer, gen_params, device)
    else:
        # Retrieve the original output
        output_stream = (data for data in output_stream_list)

    return output_stream


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


def chat_loop(
    model_path: str,
    device: str,
    num_gpus: int,
    max_gpu_memory: str,
    load_8bit: bool,
    cpu_offloading: bool,
    conv_template: Optional[str],
    temperature: float,
    repetition_penalty: float,
    max_new_tokens: int,
    chatio: ChatIO,
    gptq_config: GptqConfig,
    revision: str,
    debug: bool,
):
    # Model
    model, tokenizer = load_model(
        model_path,
        device,
        num_gpus,
        max_gpu_memory,
        load_8bit,
        cpu_offloading,
        gptq_config,
        revision,
        debug,
    )

    print(f"Model used: {type(model)}")
    print(f"Tokenizer used: {type(tokenizer)}")

    is_chatglm = "chatglm" in str(type(model)).lower()
    is_t5 = "t5" in str(type(model)).lower()
    is_falcon = "rwforcausallm" in str(type(model)).lower()

    # Hardcode T5's default repetition penalty to be 1.2
    if is_t5 and repetition_penalty == 1.0:
        repetition_penalty = 1.2

    # Chat
    def new_chat():
        if conv_template:
            conv = get_conv_template(conv_template)
        else:
            conv = get_conversation_template(model_path)
        return conv

    conv = new_chat()

    while True:
        conv = new_chat()  # reset the conversation
        try:
            inp = chatio.prompt_for_input(conv.roles[0])
        except EOFError:
            inp = ""

        if inp == "!!exit" or not inp:
            print("exit...")
            break

        if inp == "!!reset":
            print("resetting...")
            conv = new_chat()
            continue

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if is_chatglm:
            generate_stream_func = chatglm_generate_stream
        elif is_falcon:
            generate_stream_func = falcon_generate_stream
        else:
            # Use this for api call inference, otherwise use generate_stream_func = generate_stream
            generate_stream_func = generate_special_stream

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

        chatio.prompt_for_output(conv.roles[1])

        output_stream = generate_stream_func(model, tokenizer, gen_params, device)
        final_answer = chatio.stream_output(output_stream)

        t = time.time()
        duration = time.time() - t

        # Use this to append messages to the conversation history
        conv.update_last_message(final_answer.strip())

        if debug:
            num_tokens = len(tokenizer.encode(final_answer))
            msg = {
                "conv_template": conv.name,
                "prompt": prompt,
                "outputs": final_answer,
                "speed (token/s)": round(num_tokens / duration, 2),
            }
            print(f"\n{msg}\n")
