"""Download checkpoint."""
import argparse
import os

import tqdm


def run_cmd(cmd):
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="alpaca-13b-ckpt")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    files = [
        "gs://skypilot-chatbot/chatbot/13b/ckpt/added_tokens.json",
        "gs://skypilot-chatbot/chatbot/13b/ckpt/config.json",
        "gs://skypilot-chatbot/chatbot/13b/ckpt/pytorch_model-00001-of-00006.bin",
        "gs://skypilot-chatbot/chatbot/13b/ckpt/pytorch_model-00002-of-00006.bin",
        "gs://skypilot-chatbot/chatbot/13b/ckpt/pytorch_model-00003-of-00006.bin",
        "gs://skypilot-chatbot/chatbot/13b/ckpt/pytorch_model-00004-of-00006.bin",
        "gs://skypilot-chatbot/chatbot/13b/ckpt/pytorch_model-00005-of-00006.bin",
        "gs://skypilot-chatbot/chatbot/13b/ckpt/pytorch_model-00006-of-00006.bin",
        "gs://skypilot-chatbot/chatbot/13b/ckpt/pytorch_model.bin.index.json",
        "gs://skypilot-chatbot/chatbot/13b/ckpt/special_tokens_map.json",
        "gs://skypilot-chatbot/chatbot/13b/ckpt/tokenizer.model",
        "gs://skypilot-chatbot/chatbot/13b/ckpt/tokenizer_config.json",
    ]

    for filename in tqdm.tqdm(files):
        run_cmd(f"gsutil cp {filename} {output_dir}")
