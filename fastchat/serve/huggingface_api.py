"""
Usage:
python3 -m fastchat.serve.huggingface_api --model ~/model_weights/vicuna-7b/
"""
import argparse
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from fastchat.conversation import get_default_conv_template
from fastchat.serve.inference import load_model, add_model_args


@torch.inference_mode()
def main(args):
    model, tokenizer = load_model(
        args.model_path,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
        args.load_8bit,
        args.cpu_offloading,
        debug=args.debug,
    )

    msg = args.message

    conv = get_default_conv_template(args.model_path).copy()
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer([prompt]).input_ids
    output_ids = model.generate(
        torch.as_tensor(input_ids).cuda(),
        do_sample=True,
        temperature=0.7,
        max_new_tokens=1024,
    )
    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(input_ids[0]):]
    outputs = tokenizer.decode(output_ids, skip_special_tokens=True,
                               spaces_between_special_tokens=False)

    print(f"{conv.roles[0]}: {msg}")
    print(f"{conv.roles[1]}: {outputs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--message", type=str, default="Hello! Who are you?")
    args = parser.parse_args()

    main(args)
