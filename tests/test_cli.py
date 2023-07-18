"""Test command line interface for model inference."""
import argparse
import os

from fastchat.utils import run_cmd


def test_single_gpu():
    models = [
        "lmsys/vicuna-7b-v1.3",
        "lmsys/longchat-7b-16k",
        "lmsys/fastchat-t5-3b-v1.0",
        "THUDM/chatglm-6b",
        "THUDM/chatglm2-6b",
        "mosaicml/mpt-7b-chat",
        "project-baize/baize-v2-7b",
        "h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b",
        "tiiuae/falcon-7b-instruct",
        "~/model_weights/alpaca-7b",
        "~/model_weights/RWKV-4-Raven-7B-v11x-Eng99%-Other1%-20230429-ctx8192.pth",
    ]

    for model_path in models:
        if "model_weights" in model_path and not os.path.exists(
            os.path.expanduser(model_path)
        ):
            continue
        cmd = (
            f"python3 -m fastchat.serve.cli --model-path {model_path} "
            f"--style programmatic < test_cli_inputs.txt"
        )
        ret = run_cmd(cmd)
        if ret != 0:
            return

        print("")


def test_multi_gpu():
    models = [
        "lmsys/vicuna-13b-v1.3",
    ]

    for model_path in models:
        cmd = (
            f"python3 -m fastchat.serve.cli --model-path {model_path} "
            f"--style programmatic --num-gpus 2 < test_cli_inputs.txt"
        )
        ret = run_cmd(cmd)
        if ret != 0:
            return
        print("")


def test_8bit():
    models = [
        "lmsys/vicuna-13b-v1.3",
    ]

    for model_path in models:
        cmd = (
            f"python3 -m fastchat.serve.cli --model-path {model_path} "
            f"--style programmatic --load-8bit < test_cli_inputs.txt"
        )
        ret = run_cmd(cmd)
        if ret != 0:
            return
        print("")


def test_hf_api():
    models = [
        "lmsys/vicuna-7b-v1.3",
        "lmsys/fastchat-t5-3b-v1.0",
    ]

    for model_path in models:
        cmd = f"python3 -m fastchat.serve.huggingface_api --model-path {model_path}"
        ret = run_cmd(cmd)
        if ret != 0:
            return
        print("")


if __name__ == "__main__":
    test_single_gpu()
    test_multi_gpu()
    test_8bit()
    test_hf_api()
