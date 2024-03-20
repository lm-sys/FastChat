import warnings
from typing import Optional, Tuple

import torch
from flash_attn import __version__ as flash_attn_version
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_kvpacked_func,
)
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaModel,
    rotate_half,
)


def apply_rotary_pos_emb(q, k, cos_sin, position_ids):
    gather_indices = position_ids[:, :, None, None]  # [bsz, seq_len, 1, 1]
    gather_indices = gather_indices.repeat(
        1, 1, cos_sin[0].shape[1], cos_sin[0].shape[3]
    )
    bsz = gather_indices.shape[0]
    cos, sin = (
        torch.gather(x.transpose(1, 2).repeat(bsz, 1, 1, 1), 1, gather_indices)
        for x in cos_sin
    )
    q, k = ((x * cos) + (rotate_half(x) * sin) for x in (q, k))
    return q, k


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size()
    kv_heads = getattr(self, "num_key_value_heads", self.num_heads)

    q, k, v = (
        op(hidden_states).view(bsz, q_len, nh, self.head_dim)
        for op, nh in (
            (self.q_proj, self.num_heads),
            (self.k_proj, kv_heads),
            (self.v_proj, kv_heads),
        )
    )
    # shape: (b, s, num_heads, head_dim)

    kv_seq_len = k.shape[1]
    past_kv_len = 0
    if past_key_value is not None:
        past_kv_len = past_key_value[0].shape[2]
        kv_seq_len += past_kv_len

    cos_sin = self.rotary_emb(v, seq_len=kv_seq_len)
    q, k = apply_rotary_pos_emb(q, k, cos_sin, position_ids)

    if past_key_value is not None:
        assert (
            flash_attn_version >= "2.1.0"
        ), "past_key_value support requires flash-attn >= 2.1.0"
        # reuse k, v
        k = torch.cat([past_key_value[0].transpose(1, 2), k], dim=1)
        v = torch.cat([past_key_value[1].transpose(1, 2), v], dim=1)

    past_key_value = (k.transpose(1, 2), v.transpose(1, 2)) if use_cache else None

    if attention_mask is None:
        output = flash_attn_func(q, k, v, 0.0, softmax_scale=None, causal=True).view(
            bsz, q_len, -1
        )
    else:
        q, indices, cu_q_lens, max_s = unpad_input(q, attention_mask[:, -q_len:])
        # We can skip concat and call unpad twice but seems better to call unpad only once.
        kv, _, cu_k_lens, max_k = unpad_input(
            torch.stack((k, v), dim=2), attention_mask
        )
        output_unpad = flash_attn_varlen_kvpacked_func(
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
        output_unpad = output_unpad.reshape(-1, self.num_heads * self.head_dim)
        output = pad_input(output_unpad, indices, bsz, q_len)

    return self.o_proj(output), None, past_key_value


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
