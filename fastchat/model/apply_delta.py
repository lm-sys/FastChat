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

from huggingface_hub import snapshot_download
import torch
from tqdm import tqdm
import numpy as np
import git
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
    try:
        for file_path in tqdm(files):
            state_dict = torch.load(file_path)
            new_state_dict = {}

            current_size = 0
            for name, param in state_dict.items():
                param_size = param.numel() * param.element_size()

                if current_size + param_size > split_size:
                    new_file_name = f"pytorch_model-{part}.bin"
                    new_file_path = os.path.join(tmp_path, new_file_name)
                    torch.save(new_state_dict, new_file_path)
                    current_size = 0
                    new_state_dict = None
                    gc.collect()
                    new_state_dict = {}
                    part += 1

                new_state_dict[name] = param
                current_size += param_size

            new_file_name = f"pytorch_model-{part}.bin"
            new_file_path = os.path.join(tmp_path, new_file_name)
            torch.save(new_state_dict, new_file_path)
            new_state_dict = None
            gc.collect()
            new_state_dict = {}
            part += 1
    except Exception as e:
        print(f"An error occurred during split_files: {e}")
        shutil.rmtree(tmp_path)
        raise

def download_deltas(delta_path, work_dir):
    # Clone the repository without downloading LFS files
    with tqdm(total=3, desc="Cloning deltas") as pbar:
        repo = git.Repo.clone_from(f"https://huggingface.co/{delta_path}", work_dir, depth=1, no_checkout=True)
        pbar.update(1)

        # Initialize the LFS filter
        repo.git.checkout('HEAD', force=True)
        pbar.update(1)

        # Download the LFS files
        repo.git.lfs('pull', '--include="*"')
        pbar.update(1)

    return work_dir


def apply_delta(base_model_path, target_model_path, delta_path):
    downloaded_deltas = False
    deltas_work_dir = delta_path

    if not os.path.exists(delta_path):
        deltas_work_dir = os.path.join(target_model_path, "downloaded_deltas")
        os.makedirs(deltas_work_dir, exist_ok=True)
        downloaded_deltas = True
        delta_path = download_deltas(delta_path, deltas_work_dir)

        if delta_path is None:
            print("Invalid path for deltas. Exiting.")
            return

    for file in os.listdir(delta_path):
        if not file.startswith("."):
            original_file = os.path.join(base_model_path, file)
            deltas = os.path.join(delta_path, file)
            new_file = os.path.join(target_model_path, file)

            buffer_size = 4096 * 1024
            
            if os.path.exists(new_file):
                os.remove(new_file)

            with open(original_file, 'rb') as orig_stream, open(deltas, 'rb') as delta_stream, open(new_file, 'wb') as new_stream:
                while True:
                    orig_buffer = orig_stream.read(buffer_size)
                    delta_buffer = delta_stream.read(buffer_size)

                    if not delta_buffer:
                        break

                    orig_buffer_np = np.frombuffer(orig_buffer, dtype=np.uint8)
                    delta_buffer_np = np.frombuffer(delta_buffer, dtype=np.uint8)

                    if len(orig_buffer_np) < len(delta_buffer_np):
                        orig_buffer_np = np.pad(orig_buffer_np, (0, len(delta_buffer_np) - len(orig_buffer_np)), mode='constant')

                    
                    min_length = min(len(orig_buffer_np), len(delta_buffer_np))
                    new_buffer_np = np.add(delta_buffer_np[:min_length], orig_buffer_np[:min_length], dtype=np.int16)
                    
                    new_buffer_np %= 256
                    new_buffer_np = new_buffer_np.astype(np.uint8)

                    if len(delta_buffer_np) > len(orig_buffer_np):
                        new_buffer_np = np.concatenate((new_buffer_np, delta_buffer_np[len(orig_buffer_np):]))

                    new_stream.write(new_buffer_np.tobytes())


    if downloaded_deltas:
        shutil.rmtree(deltas_work_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)
    args = parser.parse_args()
    apply_delta(args.base_model_path, args.target_model_path, args.delta_path)