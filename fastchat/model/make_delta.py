"""
Make the delta weights by subtracting base weights.

Usage:
python3 -m fastchat.model.make_delta --base ~/model_weights/llama-13b --target ~/model_weights/vicuna-13b --delta ~/model_weights/vicuna-13b-delta --hub-repo-id lmsys/vicuna-13b-delta-v1.1
"""

import os
import argparse
import numpy as np

def create_deltas(original_file, new_file, delta_file):
    buffer_size = 4096 * 1024
    
    if os.path.exists(delta_file):
        os.remove(delta_file)

    with open(original_file, 'rb') as orig_stream, open(new_file, 'rb') as new_stream, open(delta_file, 'wb') as delta_stream:
        while True:
            orig_buffer = orig_stream.read(buffer_size)
            new_buffer = new_stream.read(buffer_size)

            if not new_buffer:
                break

            orig_buffer_np = np.frombuffer(orig_buffer, dtype=np.uint8)
            new_buffer_np = np.frombuffer(new_buffer, dtype=np.uint8)

            if len(orig_buffer_np) < len(new_buffer_np):
                orig_buffer_np = np.pad(orig_buffer_np, (0, len(new_buffer_np) - len(orig_buffer_np)), mode='constant')

            min_length = min(len(orig_buffer_np), len(new_buffer_np))
            delta_buffer_np = np.subtract(new_buffer_np[:min_length], orig_buffer_np[:min_length], dtype=np.int16)
            delta_buffer_np %= 256
            delta_buffer_np = delta_buffer_np.astype(np.uint8)

            if len(new_buffer_np) > len(orig_buffer_np):
                delta_buffer_np = np.concatenate((delta_buffer_np, new_buffer_np[len(orig_buffer_np):]))

            delta_stream.write(delta_buffer_np.tobytes())

def make_delta(base_model_path, target_model_path, delta_path):
    for file in os.listdir(target_model_path):
        if not file.startswith("."):
            original_file = os.path.join(base_model_path, file)
            new_file = os.path.join(target_model_path, file)
            deltas = os.path.join(delta_path, file)

            create_deltas(original_file, new_file, deltas)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)
    parser.add_argument("--hub-repo-id", type=str)
    args = parser.parse_args()

    make_delta(args.base_model_path, args.target_model_path, args.delta_path)