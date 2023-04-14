"""
Apply the delta weights on top of a base model.

Usage:
python3 -m fastchat.model.apply_delta --base ~/model_weights/llama-13b --target ~/model_weights/vicuna-13b --delta lmsys/vicuna-13b-delta-v1.1
"""
import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, AutoConfig
import gc
from torch import nn
import shutil
import os, glob

def process_weight(base_model_path,tmp_path="./tmp/"):


    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)


    split_size = 4 * 1024 * 1024 * 1024

    file_pattern = os.path.join(base_model_path, "pytorch_model-*.bin")
    files = glob.glob(file_pattern)

    part = 0

    for file_path in files:
        state_dict = torch.load(file_path, map_location=torch.device("cpu"))
        new_state_dict = {}

        current_size = 0

        for name, param in state_dict.items():
            param_size = param.numel() * param.element_size()

            if current_size + param_size > split_size:
                new_file_name = f"part{part}.bin"
                new_file_path = os.path.join(tmp_path, new_file_name)
                torch.save(new_state_dict, new_file_path)
                print(f"Saved {new_file_path}")
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
        print(f"Saved {new_file_path}")
        current_size = 0
        new_state_dict = None
        gc.collect()
        new_state_dict = {}
        part += 1

def apply_delta(base_model_path, target_model_path, delta_path):

    print(f"Loading the delta from {delta_path}")

    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

    config = AutoConfig.from_pretrained(base_model_path)

    tmp_llama_path = "./tmp1/"
    tmp_delta_path = "./tmp2/"
    
    process_weight(base_model_path,tmp_llama_path)
    process_weight(delta_path,tmp_delta_path)

    llama_pattern = os.path.join(tmp_llama_path, "part*.bin")
    llama_files = glob.glob(llama_pattern)
    delta_pattern = os.path.join(tmp_delta_path, "part*.bin")
    delta_files = glob.glob(delta_pattern)
    delta_state_dict = torch.load(delta_files[0], map_location=torch.device('cpu'))


    print("Applying the delta")
    for llama_file in llama_files:
        state_dict = torch.load(llama_file, map_location=torch.device("cpu"))
        for name, param in tqdm(state_dict.items(), desc="Applying delta"):
            if name not in delta_state_dict:
                for delta_file in delta_files:
                    delta_state_dict = torch.load(delta_file, map_location=torch.device('cpu'))
                    gc.collect()
                    if name in delta_state_dict:
                        break
                
        state_dict[name] += delta_state_dict[name]
        gc.collect()

    print(f"Saving the target model to {target_model_path}")
    base_tokenizer.save_pretrained(target_model_path)
    config.save_pretrained(target_model_path)
    shutil.copyfile(f"{base_model_path}/pytorch_model.bin.index.json", f"{target_model_path}/pytorch_model.bin.index.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)

    args = parser.parse_args()

    apply_delta(args.base_model_path, args.target_model_path, args.delta_path)
