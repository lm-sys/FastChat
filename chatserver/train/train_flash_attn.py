# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Monkey patching the LlaMA model with FlashAttn.

# Need to call this before importing transformers.
from chatserver.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()

from chatserver.train.train import train

if __name__ == "__main__":
    train()
