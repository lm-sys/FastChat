"""
Usage:
python3 -m fastchat.model.convert_fp16 --in in-folder --out out-folder
"""
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def convert_fp16(in_checkpoint, out_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(in_checkpoint, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        in_checkpoint, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.save_pretrained(out_checkpoint)
    tokenizer.save_pretrained(out_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-checkpoint", type=str, help="Path to the model")
    parser.add_argument("--out-checkpoint", type=str, help="Path to the output model")
    args = parser.parse_args()

    convert_fp16(args.in_checkpoint, args.out_checkpoint)
