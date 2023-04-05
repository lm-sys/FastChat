"""
Usage:
python3 -m fastchat.model.make_delta --base ~/model_weights/llama-13b --target ~/model_weights/vicuna-13b --delta ~/model_weights/vicuna-13b-delta --hub-repo-id lmsys/vicuna-13b-delta
"""
import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def make_delta(base_model_path, target_model_path, delta_path):
    print("Loading base model")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    print("Loading target model")
    target = AutoModelForCausalLM.from_pretrained(target_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    DEFAULT_PAD_TOKEN = "[PAD]"
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    num_new_tokens = base_tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))

    base.resize_token_embeddings(len(base_tokenizer))
    input_embeddings = base.get_input_embeddings().weight.data
    output_embeddings = base.get_output_embeddings().weight.data
    input_embeddings[-num_new_tokens:] = 0
    output_embeddings[-num_new_tokens:] = 0

    print("Calculating delta")
    for name, param in tqdm(target.state_dict().items(), desc="Calculating delta"):
        assert name in base.state_dict()
        param.data -= base.state_dict()[name]

    print("Saving delta")
    if args.hub_repo_id:
        kwargs = {"push_to_hub": True, "repo_id": args.hub_repo_id}
    else:
        kwargs = {}
    target.save_pretrained(delta_path, **kwargs)
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    target_tokenizer.save_pretrained(delta_path, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)
    parser.add_argument("--hub-repo-id", type=str, required=True)
    args = parser.parse_args()

    make_delta(args.base_model_path, args.target_model_path, args.delta_path)
