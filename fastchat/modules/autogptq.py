from dataclasses import dataclass, field
import os
from os.path import isdir, isfile
from pathlib import Path
import sys

from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM


def load_gptq_quantized_autogptq(model_dir, model_basename, seqlen):
    print("Loading AutoGPTQ quantized model...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, use_fast=True)

    # 加载量化好的模型到能被识别到的第一块显卡中
    model = AutoGPTQForCausalLM.from_quantized(
        model_dir,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device_map='auto',
        use_triton=False,
        quantize_config=None,
        device='cuda:0')

    model.seqlen = seqlen

    # 从 Hugging Face Hub 下载量化好的模型并加载到能被识别到的第一块显卡中
    # model = AutoGPTQForCausalLM.from_quantized(repo_id, device="cuda:0", use_safetensors=True, use_triton=False)

    # 使用 model.generate 执行推理
    # print(tokenizer.decode(model.generate(
    #     **tokenizer("auto_gptq is", return_tensors="pt").to(model.device))[0]))

    return model, tokenizer
