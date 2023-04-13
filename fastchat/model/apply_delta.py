"""
Apply the delta weights on top of a base model.

Usage:
python3 -m fastchat.model.apply_delta --base ~/model_weights/llama-13b --target ~/model_weights/vicuna-13b --delta lmsys/vicuna-13b-delta
"""
import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, AutoConfig
import gc
from torch import nn

def apply_delta(base_model_path, target_model_path, delta_path):

    print(f"Loading the delta from {delta_path}")
    delta_tokenizer = AutoTokenizer.from_pretrained(delta_path, use_fast=False)


    DEFAULT_PAD_TOKEN = "[PAD]"
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    num_new_tokens = base_tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))
    print("num_new_tokens: ",num_new_tokens)

    delta_num = 0
    delta_state_dict = torch.load(f"{delta_path}/pytorch_model-0000{str(delta_num%3+1)}-of-00003.bin", map_location=torch.device('cpu'))

    

    print("Applying the delta")
    for i in range(42):
        if i < 10:
            file_name = f"pytorch_model-0000{str(i)}-of-00041.bin"
        else:
            file_name = f"pytorch_model-000{str(i)}-of-00041.bin"
        state_dict_path = f"{base_model_path}/{file_name}"

        state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))

        for name, param in tqdm(state_dict.items(), desc="Applying delta"):

            while(name not in delta_state_dict):
                delta_num = (delta_num + 1) % 3
                delta_state_dict = torch.load(f"{delta_path}/pytorch_model-0000{str(delta_num%3+1)}-of-00003.bin", map_location=torch.device('cpu'))
                gc.collect()

            if param.shape != delta_state_dict[name].shape:
                new_embeddings = torch.zeros(len(base_tokenizer),state_dict[name].shape[1])
                new_embeddings[:len(base_tokenizer)-num_new_tokens] = state_dict[name]
                state_dict[name] = new_embeddings
                
            state_dict[name] += delta_state_dict[name]

        # 保存更新后的 state_dict
        torch.save(state_dict, f"{target_model_path}/{file_name}")
        state_dict = None 
        gc.collect()
        del state_dict


    print(f"Saving the target model to {target_model_path}")
    delta_tokenizer.save_pretrained(target_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)

    args = parser.parse_args()

    apply_delta(args.base_model_path, args.target_model_path, args.delta_path)
