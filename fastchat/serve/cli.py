"""
Usage:
python3 -m fastchat.serve.cli --model ~/model_weights/llama-7b
"""
import argparse
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer

from fastchat.conversation import conv_templates, SeparatorStyle

from fastchat.serve.shared_model_utils import tokensByDevice


@torch.inference_mode()
def generate_stream(tokenizer, model, params, device, debug,
                    context_len=2048, stream_interval=2):
    """Adapted from fastchat/serve/model_worker.py::generate_stream"""

    prompt = params["prompt"]
    l_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)

    if device == 'cpu-gptq':
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    elif device == 'cuda':
        input_ids = tokenizer(prompt).input_ids
    else:
        input_ids = tokenizer(prompt).input_ids

    output_ids = list(input_ids)

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]

    for i in range(max_new_tokens):
        if i == 0:
            out = model(input_ids=tokensByDevice(
                device, input_ids, True, debug, debug), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            if device == 'cuda':
                attention_mask = torch.ones(
                    1, past_key_values[0][0].shape[-2] + 1, device='cuda')
            else:
                attention_mask = torch.ones(
                    1, past_key_values[0][0].shape[-2] + 1)
            out = model(input_ids=tokensByDevice(device, token, False, debug, debug), use_cache=True,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values)
            logits = out.logits
            past_key_values = out.past_key_values

        if debug:
            print('Finished inferencing a token')

        last_token_logits = logits[0][-1]
        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if torch.is_tensor(output_ids[0]):
            output_idsPatched = [*output_ids[0].tolist(), *output_ids[1:]]
            if debug:
                print('Tokens were tensor patched for GPTQ... Tokens:',
                      output_idsPatched)
        elif type(output_ids[0]) is list:
            output_idsPatched = [*output_ids[0], *output_ids[1:]]
            if debug:
                print('Tokens were patched... Tokens:',
                      output_idsPatched)
        else:
            if debug:
                print('Tokens werent patched... Tokens:',
                      output_ids)
            output_idsPatched = output_ids

        if token == tokenizer.eos_token_id:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            output = tokenizer.decode(
                output_idsPatched, skip_special_tokens=True)
            pos = output.rfind(stop_str, l_prompt)
            if pos != -1:
                output = output[:pos]
                stopped = True
            yield output

        if stopped:
            break

    del past_key_values


def main(args):
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device
    debug = args.debug

    if device == 'cuda':
        num_gpus = int(num_gpus)
        kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "max_memory": {i: "16GiB" for i in range(num_gpus)},
        }
    elif device == 'cpu-gptq':
        kwargs = {
            "low_cpu_mem_usage": True,
            "max_memory": {0: "64GiB"},
        }
    else:
        kwargs = {
            "torch_dtype": torch.float32,
            "low_cpu_mem_usage": True,
            "max_memory": {0: "64GiB"},
        }

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f'Loading model ({device})...')
    model = AutoModelForCausalLM.from_pretrained(
        model_name, **kwargs)

    if device == 'cuda' and num_gpus == 1:
        model.cuda()

    print('Setting up convo...')
    conv = conv_templates[args.conv_template].copy()
    while True:
        try:
            inp = input(f"{conv.roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        params = {
            "model": model_name,
            "prompt": prompt,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
        }

        print(f"{conv.roles[1]}: ", end="", flush=True)
        pre = 0

        for outputs in generate_stream(tokenizer, model, params, device, debug):
            outputs = outputs[len(prompt) + 1:].strip()
            outputs = outputs.split(" ")
            now = len(outputs)
            if now - 1 > pre:
                print(" ".join(outputs[pre:now-1]), end=" ", flush=True)
                pre = now - 1
        print(" ".join(outputs[pre:]), flush=True)

        conv.messages[-1][-1] = " ".join(outputs)

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--device", type=str,
                        choices=["cuda", "cpu", "cpu-gptq"], default="cuda")
    parser.add_argument("--conv-template", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
