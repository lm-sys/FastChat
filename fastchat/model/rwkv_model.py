import os
import gc
from types import SimpleNamespace
import warnings

import torch

os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "1"

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS


class Rwkv5Tokenizer:
    def __init__(self, model):
        self.pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
        self.special_tokens_map = {"\x16": "", "\x17": ""}

    def __call__(self, x):
        if isinstance(x, list):
            input_ids = []
            for xi in x:
                input_ids.append(self.pipeline.encode(xi))
            return SimpleNamespace(input_ids=input_ids)
        return SimpleNamespace(input_ids=self.pipeline.encode(x))

    def encode(self, x, **kwargs):
        return self.pipeline.encode(x)

    def decode(self, x, **kwargs):
        return self.pipeline.decode(x)


class RwkvModel:
    def __init__(self, model_path, version="4"):
        warnings.warn(
            "Experimental support. Please use ChatRWKV if you want to chat with RWKV"
        )
        self.config = SimpleNamespace(is_encoder_decoder=False)
        self.model = RWKV(model=model_path, strategy="cuda fp16")
        # two GPUs
        # self.model = RWKV(model=model_path, strategy="cuda:0 fp16 *20 -> cuda:1 fp16")

        self.tokenizer = None
        self.version = version
        if version == "5":
            # self.tokenizer = PIPELINE(self.model, "rwkv_vocab_v20230424").tokenizer
            self.tokenizer = Rwkv5Tokenizer(self.model)
            self.pipeline = self.tokenizer.pipeline
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
        if self.version == "5":
            conv = get_conv_template("hermes-rwkv-v5")

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
        if self.version == "4":
            res_iter = generate_stream(
                self, self.tokenizer, gen_params, "cuda", context_len=4096
            )

            for res in res_iter:
                pass

            output = res["text"]
            output_ids = self.tokenizer.encode(output)
        elif self.version == "5":
            output = self.pipeline.generate(
                ctx=prompt,
                token_count=max_new_tokens,
                args=PIPELINE_ARGS(
                    temperature=max(0.2, float(temperature)),
                    top_p=0.7,
                    alpha_presence=0.1,
                    alpha_frequency=0.1,
                ),
            )
            output_ids = self.pipeline.encode(output)

        return [input_ids[0].tolist() + output_ids]


@torch.inference_mode()
def generate_stream_rwkv(
    model,
    tokenizer,
    params,
    device,
    context_len,
    stream_interval=2,
    judge_sent_end=False,
    temperature=1.0,
    top_p=0.3,
    presencePenalty=0,
    countPenalty=1.0,
):
    # __import__("ipdb").set_trace()
    ctx = params["prompt"].strip()
    # inputs = tokenizer(prompt, return_tensors="pt").to(device)
    pipeline = tokenizer.pipeline
    max_new_tokens = int(params.get("max_new_tokens", 2048))
    args = PIPELINE_ARGS(
        temperature=max(0.2, float(temperature)),
        top_p=float(top_p),
        alpha_frequency=countPenalty,
        alpha_presence=presencePenalty,
        token_ban=[],  # ban the generation of some tokens
        token_stop=[0, 24],
    )  # stop generation whenever you see any token here

    all_tokens = []
    out_last = 0
    out_str = ""
    occurrence = {}
    state = None
    for i in range(int(max_new_tokens)):
        out, state = model.model.forward(
            pipeline.encode(ctx)[-context_len:] if i == 0 else [token], state
        )
        for n in occurrence:
            out[n] -= args.alpha_presence + occurrence[n] * args.alpha_frequency
        token = pipeline.sample_logits(
            out, temperature=args.temperature, top_p=args.top_p
        )
        if token in args.token_stop:
            break
        all_tokens += [token]
        for xxx in occurrence:
            occurrence[xxx] *= 0.996
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1

        tmp = pipeline.decode(all_tokens[out_last:])
        if "\ufffd" not in tmp:
            out_str += tmp
            yield {"text": out_str.strip()}
            out_last = i + 1
    del out
    del state
    gc.collect()
    torch.cuda.empty_cache()
    yield {"text": out_str.strip()}
