import os
from types import SimpleNamespace
import warnings

import torch

os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "1"

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS


class RwkvModel:
    def __init__(self, model_path):
        warnings.warn(
            "Experimental support. Please use ChatRWKV if you want to chat with RWKV"
        )
        self.config = SimpleNamespace(is_encoder_decoder=False)
        self.model = RWKV(model=model_path, strategy="cuda fp16")
        # two GPUs
        # self.model = RWKV(model=model_path, strategy="cuda:0 fp16 *20 -> cuda:1 fp16")

        self.tokenizer = None
        self.model_path = model_path

    def to(self, target):
        assert target == "cuda"

    def __call__(self, input_ids, use_cache, past_key_values=None):
        assert use_cache == True
        input_ids = input_ids[0].detach().cpu().numpy()
        # print(input_ids)
        logits, state = self.model.forward(input_ids, past_key_values)
        # print(logits)
        logits = logits.unsqueeze(0).unsqueeze(0)
        out = SimpleNamespace(logits=logits, past_key_values=state)
        return out

    def generate(
        self, input_ids, do_sample, temperature, max_new_tokens, repetition_penalty=1.0
    ):
        # This function is used by fastchat.llm_judge.
        # Because RWKV does not support huggingface generation API,
        # we reuse fastchat.serve.inference.generate_stream as a workaround.
        from transformers import AutoTokenizer

        from fastchat.serve.inference import generate_stream
        from fastchat.conversation import get_conv_template

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/pythia-160m", use_fast=True
            )
        prompt = self.tokenizer.decode(input_ids[0].tolist())
        conv = get_conv_template("rwkv")

        gen_params = {
            "model": self.model_path,
            "prompt": prompt,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }
        res_iter = generate_stream(self, self.tokenizer, gen_params, "cuda")

        for res in res_iter:
            pass

        output = res["text"]
        output_ids = self.tokenizer.encode(output)

        return [input_ids[0].tolist() + output_ids]
