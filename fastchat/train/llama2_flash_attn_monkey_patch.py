import warnings
from typing import Optional, Tuple

import torch

is_flash_attn_2_available = False
try:
    from flash_attn import __version__ as flash_attn_version
    from flash_attn.bert_padding import pad_input, unpad_input  # type: ignore
    from flash_attn.flash_attn_interface import flash_attn_kvpacked_func, flash_attn_varlen_kvpacked_func  # type: ignore

    is_flash_attn_2_available = (
        torch.cuda.is_available() and flash_attn_version >= "2.1.0"
    )
except ImportError:
    warnings.warn("Flash Attention2 not support.")

import transformers
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaModel,
    repeat_kv,
    apply_rotary_pos_emb,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # LlamaFlashAttention2 attention does not support output_attentions
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop("padding_mask")
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )
        output_attentions = False

    bsz, q_len, _ = hidden_states.size()
    kv_heads = getattr(self, "num_key_value_heads", self.num_heads)

    q, k, v = (
        op(hidden_states).view(bsz, q_len, nh, self.head_dim).transpose(1, 2)
        for op, nh in (
            (self.q_proj, self.num_heads),
            (self.k_proj, kv_heads),
            (self.v_proj, kv_heads),
        )
    )
    # shape: (b, num_heads, s, head_dim)

    kv_seq_len = k.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
    q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    if past_key_value is not None:
        assert (
            flash_attn_version >= "2.1.0"
        ), "past_key_value support requires flash-attn >= 2.1.0"
        # reuse k, v, self_attention
        k = torch.cat([past_key_value[0], k], dim=2)
        v = torch.cat([past_key_value[1], v], dim=2)

    past_key_value = (k, v) if use_cache else None

    # cast to half precision
    input_dtype = q.dtype
    if input_dtype == torch.float32:
        # Handle the case where the model is quantized
        if hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        q = q.to(target_dtype)
        k = k.to(target_dtype)
        v = v.to(target_dtype)

    if getattr(self, "num_key_value_groups", None):
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    if attention_mask is None:
        kv = torch.stack((k, v), dim=2)
        attn_output = flash_attn_kvpacked_func(
            q, kv, 0.0, softmax_scale=None, causal=True
        )
    else:
        logger.warning_once("Padded sequences are less efficient in FlashAttention.")
        q, indices_q, cu_q_lens, max_s = unpad_input(q, attention_mask[:, -q_len:])
        # We can skip concat and call unpad twice but seems better to call unpad only once.
        kv, _, cu_k_lens, max_k = unpad_input(
            torch.stack((k, v), dim=2), attention_mask
        )
        attn_output_unpad = flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_q_lens,
            cu_k_lens,
            max_s,
            max_k,
            0.0,
            softmax_scale=None,
            causal=True,
        )
        attn_output = pad_input(attn_output_unpad, indices_q, bsz, q_len)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()

    if not output_attentions:
        attn_weights = None

    return self.o_proj(attn_output), attn_weights, past_key_value


# Disable the transformation of the attention mask in LlamaModel as flash attention
# takes a boolean key_padding_mask. Fills in the past kv length for use in forward.
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    if past_key_values_length > 0 and attention_mask is not None:
        attention_mask = torch.cat(
            (
                torch.full(
                    (input_shape[0], past_key_values_length),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
                attention_mask,
            ),
            dim=-1,
        )

    if attention_mask is not None and torch.all(attention_mask):
        return None  # This uses the faster call when training with full samples

    return attention_mask


def replace_llama_attn_with_flash_attn():
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        warnings.warn(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )

    if is_flash_attn_2_available:
        if transformers.__version__ >= "4.35.0":
            transformers.models.llama.modeling_llama.LlamaAttention = (
                transformers.models.llama.modeling_llama.LlamaFlashAttention2
            )
        else:
            LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
            LlamaAttention.forward = forward


def test():
    from fastchat.train.llama_flash_attn_monkey_patch import forward as fastchat_forward
    from transformers.models.llama.configuration_llama import LlamaConfig

    config = LlamaConfig(
        hidden_size=1024,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=8,
        max_position_embeddings=16,
    )
    device = torch.device("cuda")
    model = LlamaModel(config)
    attn = LlamaAttention(config).to(device).half()
    bsz, hs, seqlen = 2, config.hidden_size, config.max_position_embeddings
    position_ids = torch.arange(seqlen, dtype=torch.long, device=device).view(
        -1, seqlen
    )

    mask = torch.full((bsz, seqlen), True, dtype=torch.bool, device=device)
    for i in range(4):
        hidden = torch.rand((bsz, seqlen, hs), dtype=torch.float16, device=device)
        if i:
            mask[0, -i:] = False
            mask[1, :i] = False

        lmask = model._prepare_decoder_attention_mask(mask, hidden.shape[:2], hidden, 0)
        ref, _, _ = attn.forward(
            hidden, attention_mask=lmask, position_ids=position_ids
        )

        fast, _, _ = fastchat_forward(
            attn, hidden, attention_mask=mask, position_ids=position_ids
        )

        lmask = _prepare_decoder_attention_mask(
            model, mask, hidden.shape[:2], hidden, 0
        )
        test, _, _ = forward(
            attn, hidden, attention_mask=lmask, position_ids=position_ids
        )

        print(f"Mean(abs(ref)) = {torch.mean(torch.abs(ref))}")
        print(f"Mean(abs(ref - fast)) = {torch.mean(torch.abs(ref - fast))}")
        print(f"Mean(abs(ref - test)) = {torch.mean(torch.abs(ref - test))}")
        print(f"Mean(abs(fast - test)) = {torch.mean(torch.abs(fast - test))}")
        print(f"allclose(fast, test) = {torch.allclose(fast, test)}")

    with torch.no_grad():
        # Also check that past_kv is handled properly
        hidden = torch.rand((bsz, seqlen, hs), dtype=torch.float16, device=device)
        part_len = seqlen // 4
        assert part_len * 4 == seqlen
        mask = torch.full((bsz, seqlen), True, dtype=torch.bool, device=device)
        mask[0, -2:] = False
        lmask = _prepare_decoder_attention_mask(
            model, mask, hidden.shape[:2], hidden, 0
        )
        oneshot, _, _ = forward(
            attn, hidden, attention_mask=lmask, position_ids=position_ids
        )
        parts = []
        past_kv, past_kv_len = None, 0
        for i in range(4):
            start = part_len * i
            end = start + part_len
            hidden_part = hidden[:, start:end, ...]
            lmask = _prepare_decoder_attention_mask(
                model,
                mask[:, start:end],
                hidden_part.shape[:2],
                hidden_part,
                past_kv_len,
            )
            part, _, past_kv = forward(
                attn,
                hidden_part.clone(),
                attention_mask=lmask,
                position_ids=position_ids[:, start:end],
                past_key_value=past_kv,
                use_cache=True,
            )
            parts.append(part)
            past_kv_len = past_kv[0].shape[2]

        print(
            f"allclose(oneshot[:, 0], parts[0]) = {torch.allclose(oneshot[:, :part_len], parts[0])}"
        )
        print(
            f"allclose(oneshot, parts) = {torch.allclose(oneshot, torch.cat(parts, dim=1))}"
        )


if __name__ == "__main__":
    test()
