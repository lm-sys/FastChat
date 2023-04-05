import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from fastchat.conversation import conv_templates, SeparatorStyle
from fastchat.utils import disable_torch_init


@torch.inference_mode()
def main(args):
    model_name = args.model_name
    num_gpus = args.num_gpus
    max_tokens = args.max_tokens
    temp = args.temp
    useCuda = args.useCuda
    useGptq = args.useGptq

    disable_torch_init()
    if useCuda:
        kwargs = {}
    else:
        kwargs = {
            "device_map": "auto",
            "max_memory": {0: "60GiB"},
        }

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if useCuda:
        print('Loading model (CUDA)...')
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, **kwargs)
    else:
        if not useGptq:
            print('Loading model (CPU)...')
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32, **kwargs)
        else:
            print('Loading model (CPU-GPTQ)...')
            model = AutoModelForCausalLM.from_pretrained(model_name)

    if useCuda and num_gpus == 1:
        model.cuda()

    print('Setting up convo...')
    conv = conv_templates[args.conv_template].copy()
    while True:
        inp = input(f"{conv.roles[0]}: ")
        if not inp:
            print("exit...")
            break

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if useGptq:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        else:
            input_ids = tokenizer([prompt]).input_ids
        if useCuda:
            print('Generating response (CUDA)...')
            output_ids = model.generate(
                torch.as_tensor(input_ids).cuda(),
                do_sample=True,
                temperature=temp,
                max_new_tokens=max_tokens)
        else:
            print('Generating response (CPU)...')
            output_ids = model.generate(
                input_ids,
                do_sample=True,
                temperature=temp,
                max_new_tokens=max_tokens)
        print('Decoding tokens...')
        outputs = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)[0]
        sep = conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2
        try:
            index = outputs.index(sep, len(prompt))
        except ValueError:
            outputs += sep
            index = outputs.index(sep, len(prompt))
        outputs = outputs[len(prompt) + 1:index].strip()
        print(f"{conv.roles[1]}: {outputs}")
        conv.messages[-1][-1] = outputs
        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str,
                        default="facebook/opt-350m")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--conv-template", type=str, default="v1")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument('--use-cuda', dest='useCuda', action='store_true')
    parser.add_argument('--gptq', dest='useGptq', action='store_true')
    args = parser.parse_args()
    main(args)
