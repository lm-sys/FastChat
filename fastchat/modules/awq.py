from dataclasses import dataclass, field
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from awq import AutoAWQForCausalLM

@dataclass
class AWQConfig:
    ckpt: str = field(
        default=None,
        metadata={
            "help": "Load quantized model. The path to the local AWQ checkpoint."
        },
    )
    wbits: int = field(default=16, metadata={"help": "#bits to use for quantization"})
    groupsize: int = field(
        default=-1,
        metadata={"help": "Groupsize to use for quantization; default uses full row."},
    )


def load_awq_quantized(model_name, awq_config: AWQConfig, device):
    print("Loading AWQ quantized model...")
    find_awq_ckpt(awq_config)

    tokenizer = AutoTokenizer.from_pretrained(awq_config.ckpt, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_quantized(awq_config.ckpt, fuse_layers=True)

    return model.model, tokenizer


def find_awq_ckpt(awq_config: AWQConfig):
    if Path(awq_config.ckpt).is_file():
        return awq_config.ckpt

    for ext in ["*.pt", "*.safetensors","*.bin"]:
        matched_result = sorted(Path(awq_config.ckpt).glob(ext))
        if len(matched_result) > 0:
            return str(matched_result[-1])

    print("Error: AWQ checkpoint not found")
    sys.exit(1)
