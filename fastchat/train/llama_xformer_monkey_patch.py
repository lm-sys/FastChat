from typing import Optional, Tuple

import torch
import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from xformers import ops as xops


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
            Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None
    
    # Copied from modeling_open_llama.py
    attn_weights = None

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    attn_output = xops.memory_efficient_attention(
        query_states, key_states, value_states, attn_bias=xops.LowerTriangularMask(), p=0
    )
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def _get_polynomial_decay_schedule_with_warmup_lr_lambda(
      current_step: int,
      *,
      num_warmup_steps: int,
      num_training_steps: int,
      lr_end: float,
      power: float,
      lr_init: int,
  ):
      lr_end = 1e-6
      if current_step % 100 == 0:
          print("Using custom lr end {lr_end}")
      if current_step < num_warmup_steps:
          return float(current_step) / float(max(1, num_warmup_steps))
      elif current_step > num_training_steps:
          return lr_end / lr_init  # as LambdaLR multiplies by lr_init
      else:
          lr_range = lr_init - lr_end
          decay_steps = num_training_steps - num_warmup_steps
          pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
          decay = lr_range * pct_remaining**power + lr_end
          return decay / lr_init  # as LambdaLR multiplies by lr_init


def replace_llama_attn_with_xformer():
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward
    transformers.optimization._get_polynomial_decay_schedule_with_warmup_lr_lambda = _get_polynomial_decay_schedule_with_warmup_lr_lambda
