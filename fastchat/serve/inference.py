"""Inference for FastChat models."""
import abc
import gc
import math
from typing import Optional
import sys
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

from fastchat.conversation import (
    conv_templates,
    get_default_conv_template,
    SeparatorStyle,
)
from fastchat.serve.compression import load_compress_model
from fastchat.serve.monkey_patch_non_inplace import (
    replace_llama_attn_with_non_inplace_operations,
)
from fastchat.serve.serve_chatglm import chatglm_generate_stream

def raise_warning_for_incompatible_cpu_offloading_configuration(device: str, load_8bit: bool, cpu_offloading: bool):
    if cpu_offloading:
        if not load_8bit:
            warnings.warn("The cpu-offloading feature can only be used while also using 8-bit-quantization.\n"
                          "Use '--load-8bit' to enable 8-bit-quantization\n"
                          "Continuing without cpu-offloading enabled\n")
            return False
        if not "linux" in sys.platform:
            warnings.warn("CPU-offloading is only supported on linux-systems due to the limited compatability with the bitsandbytes-package\n"
                          "Continuing without cpu-offloading enabled\n")
            return False
        if device != "cuda":
            warnings.warn("CPU-offloading is only enabled when using CUDA-devices\n"
                          "Continuing without cpu-offloading enabled\n")
            return False
    return cpu_offloading

def get_gpu_memory(max_gpus=None):
    gpu_memory = []
    num_gpus = (
        torch.cuda.device_count()
        if max_gpus is None
        else min(max_gpus, torch.cuda.device_count())
    )

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory


def raise_warning_for_old_weights(model_path, model):
    if "vicuna" in model_path.lower() and isinstance(model, LlamaForCausalLM):
        if model.model.vocab_size > 32000:
            warnings.warn(
                "\nYou are probably using the old Vicuna-v0 model, "
                "which will generate unexpected results with the "
                "current fastchat.\nYou can try one of the following methods:\n"
                "1. Upgrade your weights to the new Vicuna-v1.1: https://github.com/lm-sys/FastChat#vicuna-weights.\n"
                "2. Use the old conversation template by `python3 -m fastchat.serve.cli --model-path /path/to/vicuna-v0 --conv-template conv_one_shot`\n"
                "3. Downgrade fschat to fschat==0.1.10 (Not recommonded).\n"
            )

def load_model(
    model_path, device, num_gpus, max_gpu_memory=None, load_8bit=False, cpu_offloading=False, debug=False
):
    cpu_offloading = raise_warning_for_incompatible_cpu_offloading_configuration(device, load_8bit, cpu_offloading)
    if device == "cpu":
        kwargs = {"torch_dtype": torch.float32}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus != 1:
            kwargs["device_map"] = "auto"
            if max_gpu_memory is None:
                kwargs[
                    "device_map"
                ] = "sequential"  # This is important for not the same VRAM sizes
                available_gpu_memory = get_gpu_memory(num_gpus)
                kwargs["max_memory"] = {
                    i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                    for i in range(num_gpus)
                }
            else:
                kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}
        print("init_kwargs", kwargs)
    elif device == "mps":
        kwargs = {"torch_dtype": torch.float16}
        # Avoid bugs in mps backend by not using in-place operations.
        replace_llama_attn_with_non_inplace_operations()
    else:
        raise ValueError(f"Invalid device: {device}")

    if cpu_offloading:
        # raises an error on incompatible platforms
        from transformers import BitsAndBytesConfig
        if "max_memory" in kwargs:
            kwargs["max_memory"]["cpu"] = str(math.floor(psutil.virtual_memory().available / 2**20)) + 'Mib'
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit_fp32_cpu_offload=cpu_offloading)
        kwargs["load_in_8bit"] = load_8bit
    elif load_8bit:
        if num_gpus != 1:
            warnings.warn("8-bit quantization is not supported for multi-gpu inference.")
        else:
            return load_compress_model(model_path=model_path,
                device=device, torch_dtype=kwargs["torch_dtype"])

    if "chatglm" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, **kwargs)
    elif "dolly" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
        # 50277 means "### End"
        tokenizer.eos_token_id = 50277
    elif "pythia" in model_path or "stablelm" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
    elif "t5" in model_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path,
                                                      low_cpu_mem_usage=True, **kwargs)
        tokenizer = T5Tokenizer.from_pretrained(model_path, use_fast=False)
    elif "RWKV-4" in model_path:
        from fastchat.serve.rwkv_model import RwkvModel
        model = RwkvModel(model_path)
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-160m', use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
        raise_warning_for_old_weights(model_path, model)

    if (device == "cuda" and num_gpus == 1 and not cpu_offloading) or device == "mps":
        model.to(device)

    if debug:
        print(model)

    return model, tokenizer


@torch.inference_mode()
def generate_stream(
    model, tokenizer, params, device, context_len=2048, stream_interval=2
):
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    echo = params.get("echo", True)
    stop_token_ids = params.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    input_ids = tokenizer(prompt).input_ids
    input_echo_len = len(input_ids)
    output_ids = list(input_ids)

    if model.config.is_encoder_decoder:
         max_src_len = context_len
    else:
         max_src_len = context_len - max_new_tokens - 8

    input_ids = input_ids[-max_src_len:]

    if model.config.is_encoder_decoder:
         encoder_output = model.encoder(input_ids=torch.as_tensor([input_ids],
                                                      device=device))[0]
         start_ids = torch.as_tensor([[model.generation_config.decoder_start_token_id]],
                     dtype=torch.int64, device=device)

    for i in range(max_new_tokens):
        if i == 0:
            if model.config.is_encoder_decoder:
                 out = model.decoder(input_ids=start_ids,
                                     encoder_hidden_states=encoder_output,
                                     use_cache=True)
                 logits = model.lm_head(out[0])
            else:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
            past_key_values = out.past_key_values
        else:
            if model.config.is_encoder_decoder:
                out = model.decoder(input_ids=torch.as_tensor([[token]], device=device),
                             encoder_hidden_states=encoder_output,
                             use_cache=True,
                             past_key_values=past_key_values)

                logits = model.lm_head(out[0])
            else:
                out = model(
                    input_ids=torch.as_tensor([[token]], device=device),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
            past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
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

            output = tokenizer.decode(tmp_output_ids, skip_special_tokens=True, 
                                      spaces_between_special_tokens=False)
            if stop_str:
                pos = output.rfind(stop_str, rfind_start)
                if pos != -1:
                    output = output[:pos]
                    stopped = True
            yield output

        if stopped:
            break

    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()


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
    max_new_tokens: int,
    chatio: ChatIO,
    debug: bool,
):
    # Model
    model, tokenizer = load_model(
        model_path, device, num_gpus, max_gpu_memory, load_8bit, cpu_offloading, debug
    )
    is_chatglm = "chatglm" in str(type(model)).lower()

    # Chat
    if conv_template:
        conv = conv_templates[conv_template].copy()
    else:
        conv = get_default_conv_template(model_path).copy()

    while True:
        try:
            inp = chatio.prompt_for_input(conv.roles[0])
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        if is_chatglm:
            generate_stream_func = chatglm_generate_stream
            prompt = conv.messages[conv.offset:]
        else:
            generate_stream_func = generate_stream
            prompt = conv.get_prompt()

        gen_params = {
            "model": model_path,
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }

        chatio.prompt_for_output(conv.roles[1])
        output_stream = generate_stream_func(model, tokenizer, gen_params, device)
        outputs = chatio.stream_output(output_stream)
        # NOTE: strip is important to align with the training data.
        conv.messages[-1][-1] = outputs.strip()

        if debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


def add_model_args(parser):
    parser.add_argument(
        "--model-path",
        type=str,
        default="lmsys/fastchat-t5-3b-v1.0",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda", "mps"], default="cuda",
        help="The device type"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="A single GPU like 1 or multiple GPUs like 0,2"
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="The maximum memory per gpu. Use a string like '13Gib'",
    )
    parser.add_argument(
        "--load-8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--cpu-offloading", action="store_true", help="Only when using 8-bit quantization: Offload excess weights to the CPU that don't fit on the GPU"
    )
