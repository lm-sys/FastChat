"""
Make the delta weights by subtracting base weights.

Usage:
python3 -m fastchat.model.make_delta --base ~/model_weights/llama-13b --target ~/model_weights/vicuna-13b --delta ~/model_weights/vicuna-13b-delta --hub-repo-id lmsys/vicuna-13b-delta-v1.1
"""
import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def make_delta(base_model_path, target_model_path, delta_path):
    print(f"Loading the base model from {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )

    print(f"Loading the target model from {target_model_path}")
    target = AutoModelForCausalLM.from_pretrained(
        target_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_path, use_fast=False)

    print("Calculating the delta")
    for name, param in tqdm(target.state_dict().items(), desc="Calculating delta"):
        assert name in base.state_dict()
        param.data -= base.state_dict()[name]

    print(f"Saving the delta to {delta_path}")
    if args.hub_repo_id:
        kwargs = {"push_to_hub": True, "repo_id": args.hub_repo_id}
    else:
        kwargs = {}
    target.save_pretrained(delta_path, **kwargs)
    target_tokenizer.save_pretrained(delta_path, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)
    parser.add_argument("--hub-repo-id", type=str)
    args = parser.parse_args()

    make_delta(args.base_model_path, args.target_model_path, args.delta_path)
