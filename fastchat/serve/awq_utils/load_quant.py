import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from awq.quantize.quantizer import real_quantize_model_weight
from awq.quantize.qmodule import WQLinear
from tqdm import tqdm

def load_awq_model(model, checkpoint, w_bit, group_size, device):
    q_config = {"zero_point": True, "q_group_size": group_size}
    real_quantize_model_weight(model, w_bit, q_config, init_only = True)
    pbar = tqdm(range(1))
    pbar.set_description('Loading checkpoint')
    for i in pbar:
        if hasattr(model.config, "tie_encoder_decoder"):
            model.config.tie_encoder_decoder = False
        if hasattr(model.config, "tie_word_embeddings"):
            model.config.tie_word_embeddings = False
        model = load_checkpoint_and_dispatch(
            model, checkpoint,
            no_split_module_classes=[
                "OPTDecoderLayer", "LlamaDecoderLayer", "BloomBlock", "MPTBlock", "DecoderLayer"]
        ).to(device)
    return model


def make_quant_linear(module, names, w_bit, groupsize, device, name=''):
    if isinstance(module, WQLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            delattr(module, attr)
            setattr(module, attr, WQLinear(w_bit, groupsize, tmp.in_features, tmp.out_features, tmp.bias is not None, device))
    for name1, child in module.named_children():
        make_quant_linear(child, names, w_bit, groupsize, device, name + '.' + name1 if name != '' else name1)

def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res


def load_awq_llama_fast(model, checkpoint, w_bit, group_size, device):
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    make_quant_linear(model, layers, w_bit, group_size, device)
    del layers

    pbar = tqdm(range(1))
    pbar.set_description('Loading checkpoint')
    for i in pbar:
        if checkpoint.endswith('.safetensors'):
            from safetensors.torch import load_file as safe_load
            model.load_state_dict(safe_load(checkpoint))
        else:
            model.load_state_dict(torch.load(checkpoint))

    return model.to(device)