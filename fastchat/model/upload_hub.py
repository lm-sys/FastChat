"""
Upload weights to huggingface.

Usage:
python3 -m fastchat.model.upload_hub --model-path ~/model_weights/vicuna-13b --hub-repo-id lmsys/vicuna-13b-v1.3
"""
import argparse
import tempfile

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def upload_hub(model_path, hub_repo_id):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    kwargs = {"push_to_hub": True, "repo_id": hub_repo_id}

    with tempfile.TemporaryDirectory() as tmp_path:
        model.save_pretrained(tmp_path, **kwargs)
        tokenizer.save_pretrained(tmp_path, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--hub-repo-id", type=str, required=True)
    args = parser.parse_args()

    upload_hub(args.model_path, args.hub_repo_id)
