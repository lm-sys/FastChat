"""
Download the pre-generated model answers and judgments for MT-bench.
"""
import os

from fastchat.utils import run_cmd

filenames = [
    "data/mt_bench/model_answers/alpaca-13b.jsonl",
    "data/mt_bench/model_answers/baize-v2-13b.jsonl",
    "data/mt_bench/model_answers/chatglm-6b.jsonl",
    "data/mt_bench/model_answers/claude-instant-v1.jsonl",
    "data/mt_bench/model_answers/claude-v1.jsonl",
    "data/mt_bench/model_answers/dolly-v2-12b.jsonl",
    "data/mt_bench/model_answers/falcon-40b-instruct.jsonl",
    "data/mt_bench/model_answers/fastchat-t5-3b.jsonl",
    "data/mt_bench/model_answers/gpt-3.5-turbo.jsonl",
    "data/mt_bench/model_answers/gpt-4.jsonl",
    "data/mt_bench/model_answers/gpt4all-13b-snoozy.jsonl",
    "data/mt_bench/model_answers/guanaco-33b.jsonl",
    "data/mt_bench/model_answers/guanaco-65b.jsonl",
    "data/mt_bench/model_answers/h2ogpt-oasst-open-llama-13b.jsonl",
    "data/mt_bench/model_answers/koala-13b.jsonl",
    "data/mt_bench/model_answers/llama-13b.jsonl",
    "data/mt_bench/model_answers/mpt-30b-chat.jsonl",
    "data/mt_bench/model_answers/mpt-30b-instruct.jsonl",
    "data/mt_bench/model_answers/mpt-7b-chat.jsonl",
    "data/mt_bench/model_answers/nous-hermes-13b.jsonl",
    "data/mt_bench/model_answers/oasst-sft-4-pythia-12b.jsonl",
    "data/mt_bench/model_answers/oasst-sft-7-llama-30b.jsonl",
    "data/mt_bench/model_answers/palm-2-chat-bison-001.jsonl",
    "data/mt_bench/model_answers/rwkv-4-raven-14b.jsonl",
    "data/mt_bench/model_answers/stablelm-tuned-alpha-7b.jsonl",
    "data/mt_bench/model_answers/tulu-30b.jsonl",
    "data/mt_bench/model_answers/vicuna-13b-v1.3.jsonl",
    "data/mt_bench/model_answers/vicuna-33b-v1.3.jsonl",
    "data/mt_bench/model_answers/vicuna-7b-v1.3.jsonl",
    "data/mt_bench/model_answers/wizardlm-13b.jsonl",
    "data/mt_bench/model_answers/wizardlm-30b.jsonl",

    "data/mt_bench/model_judgment/gpt-4_pair.jsonl",
    "data/mt_bench/model_judgment/gpt-4_single.jsonl",
]


if __name__ == "__main__":
    prefix = "https://huggingface.co/spaces/lmsys/mt-bench/resolve/main/"

    for name in filenames:
        os.makedirs(os.path.dirname(name), exist_ok=True)
        run_cmd(f"wget -q --show-progress -O {name} {prefix + name}")
