## Vicuna Weights

| Weights version | v1.3 | v1.1 | v0 |
| ---- | ---- | ---- | ---- |
| Link | [7B](https://huggingface.co/lmsys/vicuna-7b-v1.3), [13B](https://huggingface.co/lmsys/vicuna-13b-v1.3), [33B](//huggingface.co/lmsys/vicuna-33b-v1.3) | [7B](https://huggingface.co/lmsys/vicuna-7b-delta-v1.1), [13B](https://huggingface.co/lmsys/vicuna-13b-delta-v1.1) | [7B](https://huggingface.co/lmsys/vicuna-7b-delta-v0), [13B](https://huggingface.co/lmsys/vicuna-13b-delta-v0) |
| Separator | `</s>` | `</s>` | `###` |
| Is delta weights | No | Yes | Yes |
| FastChat PyPI package compatibility | >= v0.2.1 | >= v0.2.1 |<= v0.1.10 |
| FastChat source code compatibility | after [tag v0.2.1](https://github.com/lm-sys/FastChat/tree/v0.2.1) | after [tag v0.2.1](https://github.com/lm-sys/FastChat/tree/v0.2.1) | [tag v0.1.10](https://github.com/lm-sys/FastChat/tree/v0.1.10) |

### Updates
- Major updates of weights v1.3
  - Train with twice the amount of ShareGPT data compared to previous versions.
  - Provide merged weights directly instead of delta weights.

- Major updates of weights v1.1
  - Refactor the tokenization and separator. In Vicuna v1.1, the separator has been changed from `###` to the EOS token `</s>`. This change makes it easier to determine the generation stop criteria and enables better compatibility with other libraries.
  - Fix the supervised fine-tuning loss computation for better model quality.

## Prompt Template

### Example prompt (weights v1.1 and v1.3)
```
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: Hello!
ASSISTANT: Hello!</s>
USER: How are you?
ASSISTANT: I am good.</s>
```

See a full prompt template [here](https://github.com/lm-sys/FastChat/blob/daa2b9abe20597ebf34dc5df164d450456610c74/fastchat/conversation.py#L246-L259).

### Example prompt (weights v0)
```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.

### Human: Hello!
### Assistant: Hello!
### Human: How are you?
### Assistant: I am good.
```

See the full prompt template [here](https://github.com/lm-sys/FastChat/blob/daa2b9abe20597ebf34dc5df164d450456610c74/fastchat/conversation.py#L198-L229).

## How to Apply Delta Weights (for weights v1.1 and v0)

We release [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/) weights (v1.1 and v0) as delta weights to comply with the LLaMA model license.
You can add our delta to the original LLaMA weights to obtain the Vicuna weights. Instructions:

1. Get the original LLaMA weights in the Hugging Face format by following the instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama).
2. Use the following scripts to get Vicuna weights by applying our delta. They will automatically download delta weights from our Hugging Face [account](https://huggingface.co/lmsys).

**NOTE**:
Weights v1.1 are only compatible with ```transformers>=4.28.0``` and ``fschat >= 0.2.0``.
Please update your local packages accordingly. If you follow the above commands to do a fresh install, then you should get all the correct versions.

#### Vicuna-7B
This conversion command needs around 30 GB of CPU RAM.
See the "Low CPU Memory Conversion" section below if you do not have enough memory.
Replace `/path/to/*` with the real paths.
```bash
python3 -m fastchat.model.apply_delta \
    --base-model-path /path/to/llama-7b \
    --target-model-path /path/to/output/vicuna-7b \
    --delta-path lmsys/vicuna-7b-delta-v1.1
```

#### Vicuna-13B
This conversion command needs around 60 GB of CPU RAM.
See the "Low CPU Memory Conversion" section below if you do not have enough memory.
Replace `/path/to/*` with the real paths.
```bash
python3 -m fastchat.model.apply_delta \
    --base-model-path /path/to/llama-13b \
    --target-model-path /path/to/output/vicuna-13b \
    --delta-path lmsys/vicuna-13b-delta-v1.1
```

#### Low CPU Memory Conversion
You can try these methods to reduce the CPU RAM requirement of weight conversion.
1. Append `--low-cpu-mem` to the commands above, which will split large weight files into smaller ones and use the disk as temporary storage. This can keep the peak memory at less than 16GB.
2. Create a large swap file and rely on the operating system to automatically utilize the disk as virtual memory.

## FAQ

### Tokenizer issues
There are some frequently asked tokenizer issues (https://github.com/lm-sys/FastChat/issues/408).
Some of them are not only related to FastChat or Vicuna weights but are also related to how you convert the base llama model.

We suggest that you use `transformers>=4.28.0` and redo the weight conversion for the base llama model.
After applying the delta, you should have a file named `special_tokens_map.json` in your converted weight folder for either v0 or v1.1.
The contents of this file should be the same as this file: https://huggingface.co/lmsys/vicuna-13b-delta-v0/blob/main/special_tokens_map.json.
If the file is not present, please copy the `special_tokens_map.json` and `tokenizer_config.json` files from https://huggingface.co/lmsys/vicuna-13b-delta-v0/tree/main to your converted weight folder. This works for both v0 and v1.1.
