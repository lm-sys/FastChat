import glob
import os
import re
from accelerate import init_empty_weights
from accelerate.utils import get_balanced_memory, infer_auto_device_map
from huggingface_hub import snapshot_download
import psutil
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    BitsAndBytesConfig,
)


def load_compress_model(
    model_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float32,
    use_fast: bool = True,
    revision="main",
    num_gpus=1,
    max_gpu_memory=None,
    quant_bits=8,
    trust_remote_code=True,
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
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        except ValueError or NameError or KeyError:
            model = AutoModel.from_config(config, trust_remote_code=True)
        if device not in ["cpu", "mps"]:
            _max_memory = get_balanced_memory(
                model,
                dtype=torch_dtype,
                low_zero=False,
                no_split_module_classes=model._no_split_modules,
            )
            if max_gpu_memory:
                gbit = int(
                    float(re.search("[0-9]+", max_gpu_memory).group(0)) * 1024**3
                )
                max_memory = {i: gbit for i in range(num_gpus)}
                max_memory["cpu"] = psutil.virtual_memory().available
                #  gpu memory must be the minimum of allowed and inferred
                for key, value in max_memory.items():
                    max_memory[key] = min(value, _max_memory.get(key, value))
            else:
                max_memory = _max_memory
            device_map = infer_auto_device_map(
                model,
                dtype=torch_dtype,
                max_memory=max_memory,
                no_split_module_classes=model._no_split_modules,
            )
        else:
            device_map = None
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
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=quant_bits == 4,
        load_in_8bit=quant_bits == 8,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=False,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,
            load_in_4bit=quant_bits == 4,
            load_in_8bit=quant_bits == 8,
            device_map=device_map,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
    except ValueError or KeyError or NameError:
        model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_path,
            load_in_4bit=quant_bits == 4,
            load_in_8bit=quant_bits == 8,
            device_map=device_map,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
    model.eval()
    print("Loading and compressing model done.")
    return model, tokenizer
