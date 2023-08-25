from dataclasses import dataclass, field
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, modeling_utils


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

    try:
        from tinychat.utils import load_quant
        from tinychat.modules import make_quant_norm, make_quant_attn, make_fused_mlp
    except ImportError as e:
        print(f"Error: Failed to import tinychat. {e}")
        print("Please double check if you have successfully installed AWQ")
        print("See https://github.com/lm-sys/FastChat/blob/main/docs/awq.md")
        sys.exit(-1)

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=False, trust_remote_code=True
    )

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.kaiming_normal_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    modeling_utils._init_weights = False

    torch.set_default_dtype(torch.half)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    if any(name in find_awq_ckpt(awq_config) for name in ["llama", "vicuna"]):
        model = load_quant.load_awq_llama_fast(
            model,
            find_awq_ckpt(awq_config),
            awq_config.wbits,
            awq_config.groupsize,
            device,
        )
        make_quant_attn(model, device)
        make_quant_norm(model)
        make_fused_mlp(model)
    else:
        model = load_quant.load_awq_model(
            model,
            find_awq_ckpt(awq_config),
            awq_config.wbits,
            awq_config.groupsize,
            device,
        )
    return model, tokenizer


def find_awq_ckpt(awq_config: AWQConfig):
    if Path(awq_config.ckpt).is_file():
        return awq_config.ckpt

    for ext in ["*.pt", "*.safetensors"]:
        matched_result = sorted(Path(awq_config.ckpt).glob(ext))
        if len(matched_result) > 0:
            return str(matched_result[-1])

    print("Error: AWQ checkpoint not found")
    sys.exit(1)
