import os
from types import SimpleNamespace
import warnings

import torch

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1'

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS


class RwkvModel:
    def __init__(self, model_path):
        warnings.warn("Experimental support. Please use ChatRWKV if you want to chat with RWKV")
        self.config = SimpleNamespace(is_encoder_decoder=False)
        self.model = RWKV(model=model_path, strategy='cuda fp16')

    def to(self, target):
        assert target == 'cuda'

    def __call__(self, input_ids, use_cache, past_key_values=None):
        assert use_cache == True
        input_ids = input_ids[0].detach().cpu().numpy()
        # print(input_ids)
        logits, state = self.model.forward(input_ids, past_key_values)
        # print(logits)
        out = SimpleNamespace(logits = [[logits]], past_key_values=state)
        return out
