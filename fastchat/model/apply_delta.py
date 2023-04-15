"""
Apply the delta weights on top of a base model.

Usage:
python3 -m fastchat.model.apply_delta --base ~/model_weights/llama-7b --target ~/model_weights/vicuna-7b --delta lmsys/vicuna-7b-delta-v1.1
"""
import argparse
import gc
import glob
import os
import shutil
import json

from huggingface_hub import snapshot_download
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm


GB = 1 << 30


def split_files(model_path, tmp_path, split_size):
    if not os.path.exists(model_path):
        model_path = snapshot_download(repo_id=model_path)
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    file_pattern = os.path.join(model_path, "pytorch_model-*.bin")
    files = glob.glob(file_pattern)

    part = 0
    for file_path in tqdm(files):
        state_dict = torch.load(file_path)
        new_state_dict = {}

        current_size = 0

        for name, param in state_dict.items():
            param_size = param.numel() * param.element_size()

            if current_size + param_size > split_size:
                new_file_name = f"part{part}.bin"
                new_file_path = os.path.join(tmp_path, new_file_name)
                torch.save(new_state_dict, new_file_path)
                current_size = 0
                new_state_dict = None
                gc.collect()
                new_state_dict = {}
                part += 1

            new_state_dict[name] = param
            current_size += param_size

        new_file_name = f"part{part}.bin"
        new_file_path = os.path.join(tmp_path, new_file_name)
        torch.save(new_state_dict, new_file_path)
        new_state_dict = None
        gc.collect()
        new_state_dict = {}
        part += 1


def apply_delta(base_model_path, target_model_path, delta_path):
    print(f"Loading the delta from {delta_path}")

    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    base_config = AutoConfig.from_pretrained(base_model_path)

    tmp_base_path = "tmp1/"
    tmp_delta_path = "tmp2/"
    split_size = 4 * GB

    print(f"Split files for the base model")
    split_files(base_model_path, tmp_base_path, split_size)
    print(f"Split files for the delta model")
    split_files(delta_path, tmp_delta_path, split_size)

    base_pattern = os.path.join(tmp_base_path, "part*.bin")
    base_files = glob.glob(base_pattern)
    delta_pattern = os.path.join(tmp_delta_path, "part*.bin")
    delta_files = glob.glob(delta_pattern)
    delta_state_dict = torch.load(delta_files[0])

    print("Applying the delta")
    weight_map = {}
    total_size = 0

    for i , base_file in tqdm(enumerate(base_files)):
        state_dict = torch.load(base_file)
        file_name = f"part{i}.bin"
        for name, param in state_dict.items():
            if name not in delta_state_dict:
                for delta_file in delta_files:
                    delta_state_dict = torch.load(delta_file)
                    gc.collect()
                    if name in delta_state_dict:
                        break

            state_dict[name] += delta_state_dict[name]
            weight_map[name] = file_name
            total_size += param.numel() * param.element_size()
            gc.collect()
        torch.save(state_dict,os.path.join(target_model_path, file_name))

    with open(os.path.join(target_model_path,"pytorch_model.bin.index.json"), "w") as f:
        json.dump({"weight_map":weight_map,"metadata":{"total_size":total_size}}, f)
        
    print(f"Saving the target model to {target_model_path}")
    base_tokenizer.save_pretrained(target_model_path)
    base_config.save_pretrained(target_model_path)
    shutil.rmtree(tmp_base_path)
    shutil.rmtree(tmp_delta_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)

    args = parser.parse_args()

    apply_delta(args.base_model_path, args.target_model_path, args.delta_path)
