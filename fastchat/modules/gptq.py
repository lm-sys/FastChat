import os
import sys
from dataclasses import dataclass, field
from os.path import isdir, isfile
from pathlib import Path

from transformers import AutoTokenizer

script_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
module_path = os.path.join(script_path, "../repositories/GPTQ-for-LLaMa")

try:
    sys.path.insert(0, module_path)
    from llama import load_quant
except ImportError:
    print("Error: Failed to load GPTQ-for-LLaMa")
    print("See https://github.com/lm-sys/FastChat/blob/main/docs/gptq.md")
    sys.exit(-1)


@dataclass
class GptqConfig:
    ckpt: str = field(
        default=None,
        metadata={
            "help": "Load quantized model. The path to the local GPTQ checkpoint."
        },
    )
    wbits: int = field(default=16, metadata={"help": "#bits to use for quantization"})
    groupsize: int = field(
        default=-1,
        metadata={"help": "Groupsize to use for quantization; default uses full row."},
    )
    act_order: bool = field(
        default=True,
        metadata={"help": "Whether to apply the activation order GPTQ heuristic"},
    )


def load_gptq_quantized(model_name, gptq_config: GptqConfig, device):
    print("Loading GPTQ quantized model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = load_quant(
        model_name,
        find_gptq_ckpt(gptq_config),
        gptq_config.wbits,
        gptq_config.groupsize,
        act_order=gptq_config.act_order,
    )
    model.to(device)

    return model, tokenizer


def find_gptq_ckpt(gptq_config: GptqConfig):
    if Path(gptq_config.ckpt).is_file():
        return gptq_config.ckpt

    for ext in ["*.pt", "*.safetensors"]:
        matched_result = sorted(Path(gptq_config.ckpt).glob(ext))
        if len(matched_result) > 0:
            return str(matched_result[-1])

    print("Error: gptq checkpoint not found")
    sys.exit(1)
