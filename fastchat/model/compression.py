import dataclasses
import gc
import glob
import os
import re

from accelerate import init_empty_weights
from accelerate.utils import (
    set_module_tensor_to_device,
    get_balanced_memory,
    infer_auto_device_map,
)
from huggingface_hub import snapshot_download
import psutil
import torch
from torch import Tensor
from torch.nn import functional as F
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

@dataclasses.dataclass
class CompressionConfig:
    """Group-wise quantization."""

    num_bits: int
    group_size: int
    group_dim: int
    symmetric: bool
    enabled: bool = True


default_compression_config = CompressionConfig(
    num_bits=8, group_size=256, group_dim=1, symmetric=True, enabled=True
)


class CLinear(nn.Module):
    """Compressed Linear Layer."""

    def __init__(self, weight=None, bias=None, device=None):
        super().__init__()
        if weight is None:
            self.weight = None
        elif isinstance(weight, Tensor):
            self.weight = compress(weight.data, default_compression_config)
        else:
            self.weight = weight
        self.bias = bias
        self.device = device

    def forward(self, input: Tensor) -> Tensor:
        weight = decompress(self.weight, default_compression_config)
        input = input.to(weight.dtype).to(self.device)
        if self.bias is None:
            return F.linear(input, weight)
        self.bias = self.bias.to(weight.dtype).to(self.device)
        return F.linear(input, weight, self.bias)


# unused function
def compress_module(module, target_device):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.Linear:
            setattr(
                module,
                attr_str,
                CLinear(target_attr.weight, target_attr.bias, target_device),
            )
    for name, child in module.named_children():
        compress_module(child, target_device)


def get_compressed_list(module, prefix=""):
    compressed_list = []
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.Linear:
            full_name = (
                f"{prefix}.{attr_str}.weight" if prefix else f"{attr_str}.weight"
            )
            compressed_list.append(full_name)
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        for each in get_compressed_list(child, child_prefix):
            compressed_list.append(each)
    return compressed_list


def apply_compressed_weight(module, compressed_state_dict, prefix=""):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.Linear:
            # set every Linear to be CLinear
            full_name = (
                f"{prefix}.{attr_str}.weight" if prefix else f"{attr_str}.weight"
            )
            setattr(
                module,
                attr_str,
                CLinear(
                    compressed_state_dict[full_name],
                    target_attr.bias,
                    compressed_state_dict[full_name][0].device,
                ),
            )
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        apply_compressed_weight(child, compressed_state_dict, prefix=child_prefix)


def load_compress_model(
    model_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float32,
    use_fast: bool = True,
    revision="main",
    num_gpus=1,
    max_gpu_memory=None,
):
    print("Loading and compressing model...")
    # partially load model
    # `use_fast=True`` is not supported for some models.
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=use_fast, revision=revision, trust_remote_code=True
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=~use_fast, revision=revision, trust_remote_code=True
        )
    with init_empty_weights():
        # `trust_remote_code` should be set as `True` for both AutoConfig and AutoModel
        config = AutoConfig.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
            revision=revision,
            trust_remote_code=True,
        )
        # some models are loaded by AutoModel but not AutoModelForCausalLM,
        # such as chatglm, chatglm2
        try:
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True
            )
        except NameError:
            model = AutoModel.from_config(config, trust_remote_code=True)
        if device == "cuda":
            if not max_gpu_memory:
                max_memory = get_balanced_memory(
                    model,
                    dtype=torch_dtype,
                    low_zero=False,
                    no_split_module_classes=model._no_split_modules,
                )
            else:
                gbit = int(
                    float(re.search("[0-9]+", max_gpu_memory).group(0)) * 1024**3
                )
                max_memory = {i: gbit for i in range(num_gpus)}
                max_memory["cpu"] = psutil.virtual_memory().available
            device_map = infer_auto_device_map(
                model,
                dtype=torch_dtype,
                max_memory=max_memory,
                no_split_module_classes=model._no_split_modules,
            )
        else:
            device_map = None
        linear_weights = get_compressed_list(model)
    if os.path.exists(model_path):
        # `model_path` is a local folder
        base_pattern = os.path.join(model_path, "pytorch_model*.bin")
    else:
        # `model_path` is a cached Hugging Face repo
        # We don't necessarily need to download the model' repo again if there is a cache.
        # So check the default huggingface cache first.
        model_path_temp = os.path.join(
            os.getenv("HOME"),
            ".cache/huggingface/hub",
            "models--" + model_path.replace("/", "--"),
            "snapshots/",
        )
        downloaded = False
        if os.path.exists(model_path_temp):
            temp_last_dir = os.listdir(model_path_temp)[-1]
            model_path_temp = os.path.join(model_path_temp, temp_last_dir)
            base_pattern = os.path.join(model_path_temp, "pytorch_model*.bin")
            files = glob.glob(base_pattern)
            if len(files) > 0:
                downloaded = True

        if downloaded:
            model_path = model_path_temp
        else:
            model_path = snapshot_download(model_path, revision=revision)
        base_pattern = os.path.join(model_path, "pytorch_model*.bin")
    files = glob.glob(base_pattern)
    if len(files) == 0:
        raise ValueError(
            f"Cannot find any model weight files. "
            f"Please check your (cached) weight path: {model_path}"
        )
    compressed_state_dict = {}
    for filename in tqdm(files):
        tmp_state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        for name in tmp_state_dict:
            device_rank = get_sublayer_device(
                device_map=device_map, layer_name=name, device=device
            )
            if name in linear_weights:
                # send linear_weights to corresponding device rank,such as CUDA:0
                tensor = tmp_state_dict[name].to(device_rank).data.to(torch_dtype)
                # execute compress on the device rank
                compressed_state_dict[name] = compress(
                    tensor, default_compression_config
                )
            else:
                compressed_state_dict[name] = tmp_state_dict[name].to(device_rank)
            tmp_state_dict[name] = None
            tensor = None
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            if device == "xpu":
                torch.xpu.empty_cache()

    for name in model.state_dict():
        if name not in linear_weights:
            device_rank = get_sublayer_device(
                device_map=device_map, layer_name=name, device=device
            )
            set_module_tensor_to_device(
                model, name, device_rank, value=compressed_state_dict[name]
            )
    apply_compressed_weight(model, compressed_state_dict)

    if torch_dtype == torch.float16:
        model.half()
    # model.to(device)
    model.eval()
    print("Loading and compressing model done.")
    return model, tokenizer


def compress(tensor, config):
    """Simulate group-wise quantization."""
    if not config.enabled:
        return tensor

    group_size, num_bits, group_dim, symmetric = (
        config.group_size,
        config.num_bits,
        config.group_dim,
        config.symmetric,
    )
    assert num_bits <= 8

    original_shape = tensor.shape
    num_groups = (original_shape[group_dim] + group_size - 1) // group_size
    new_shape = (
        original_shape[:group_dim]
        + (num_groups, group_size)
        + original_shape[group_dim + 1 :]
    )

    # Pad
    pad_len = (group_size - original_shape[group_dim] % group_size) % group_size
    if pad_len != 0:
        pad_shape = (
            original_shape[:group_dim] + (pad_len,) + original_shape[group_dim + 1 :]
        )
        tensor = torch.cat(
            [tensor, torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)],
            dim=group_dim,
        )
    data = tensor.view(new_shape)

    # Quantize
    if symmetric:
        B = 2 ** (num_bits - 1) - 1
        scale = B / torch.max(data.abs(), dim=group_dim + 1, keepdim=True)[0]
        data = data * scale
        data = data.clamp_(-B, B).round_().to(torch.int8)
        return data, scale, original_shape
    else:
        B = 2**num_bits - 1
        mn = torch.min(data, dim=group_dim + 1, keepdim=True)[0]
        mx = torch.max(data, dim=group_dim + 1, keepdim=True)[0]

        scale = B / (mx - mn)
        data = data - mn
        data.mul_(scale)

        data = data.clamp_(0, B).round_().to(torch.uint8)
        return data, mn, scale, original_shape


def decompress(packed_data, config):
    """Simulate group-wise dequantization."""
    if not config.enabled:
        return packed_data

    group_size, num_bits, group_dim, symmetric = (
        config.group_size,
        config.num_bits,
        config.group_dim,
        config.symmetric,
    )

    # Dequantize
    if symmetric:
        data, scale, original_shape = packed_data
        data = data / scale
    else:
        data, mn, scale, original_shape = packed_data
        data = data / scale
        data.add_(mn)

    # Unpad
    pad_len = (group_size - original_shape[group_dim] % group_size) % group_size
    if pad_len:
        padded_original_shape = (
            original_shape[:group_dim]
            + (original_shape[group_dim] + pad_len,)
            + original_shape[group_dim + 1 :]
        )
        data = data.reshape(padded_original_shape)
        indices = [slice(0, x) for x in original_shape]
        return data[indices].contiguous()
    else:
        return data.view(original_shape)


def get_sublayer_device(device_map: dict, layer_name: str, device: str):
    if device == "cpu":
        return "cpu"
    for key, value in device_map.items():
        if key in layer_name:
            return f"{device}:{value}"


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    model, tokenizer = load_compress_model(
        model_path="THUDM/chatglm2-6b",
        device="cuda",
        torch_dtype=torch.float16,
        use_fast=True,
        revision="main",
        num_gpus=2,
        max_gpu_memory="20GiB",
    )
    input_ids = tokenizer.encode("hello there")
    res = model.generate(input_ids)
    print("done!")
