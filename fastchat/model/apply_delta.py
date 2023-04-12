"""
Apply the delta weights on top of a base model.

Usage:
python3 -m fastchat.model.apply_delta --base ~/model_weights/llama-13b --target ~/model_weights/vicuna-13b --delta lmsys/vicuna-13b-delta-v1.1
"""
import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def apply_delta(base_model_path, target_model_path, delta_path):
    print(f"Loading the base model from {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    base_tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, use_fast=False)

    print(f"Loading the delta from {delta_path}")
    delta = AutoModelForCausalLM.from_pretrained(delta_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    print("Applying the delta")
    for name, param in tqdm(base.state_dict().items(), desc="Applying delta"):
        assert name in delta.state_dict()
        param.data += delta.state_dict()[name]

    print(f"Saving the target model to {target_model_path}")
    base.save_pretrained(target_model_path)
    base_tokenizer.save_pretrained(target_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)

    args = parser.parse_args()

    apply_delta(args.base_model_path, args.target_model_path, args.delta_path)
