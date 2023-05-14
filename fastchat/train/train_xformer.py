# Make it more memory efficient by monkey patching the LLaMA model with XFormer's
# memory-efficient attention.

# Need to call this before importing transformers.
from fastchat.train.llama_xformer_monkey_patch import (
    replace_llama_attn_with_xformer
)

if __name__ == "__main__":
    replace_llama_attn_with_xformer()
    from fastchat.train.train import train
    train()
