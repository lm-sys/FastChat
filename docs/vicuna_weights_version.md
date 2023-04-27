## Vicuna-7B/13B

| Weights version | v1.1 | v0 |
| ---- | ---- | ---- |
| Link      | [7B](https://huggingface.co/lmsys/vicuna-7b-delta-v1.1), [13B](https://huggingface.co/lmsys/vicuna-13b-delta-v1.1) | [7B](https://huggingface.co/lmsys/vicuna-7b-delta-v0), [13B](https://huggingface.co/lmsys/vicuna-13b-delta-v0) |
| Separator | `</s>` | `###` |
| FastChat PyPI package compatibility   | >= v0.2.1 |<= v0.1.10 |
| FastChat source code compatibility | after [tag v0.2.1](https://github.com/lm-sys/FastChat/tree/v0.2.1) | [tag v0.1.10](https://github.com/lm-sys/FastChat/tree/v0.1.10) |

Major updates of weights v1.1
- Refactor the tokenization and separator. In Vicuna v1.1, the separator has been changed from `###` to the EOS token `</s>`. This change makes it easier to determine the generation stop criteria and enables better compatibility with other libraries.
- Fix the supervised fine-tuning loss computation for better model quality.

### Example prompt (Weight v1.1)
```
A chat between a user and an assistant.

USER: Hello!
ASSISTANT: Hello!</s>
USER: How are you?
ASSISTANT: I am good.</s>
```

See a full prompt template [here](https://github.com/lm-sys/FastChat/blob/00d9e6675bdff60be6603ffff9313b1d797d2e3e/fastchat/conversation.py#L115-L124).

### Example prompt (Weight v0)
```
A chat between a human and an assistant.

### Human: Hello!
### Assistant: Hello!
### Human: How are you?
### Assistant: I am good.
```

See the full prompt template [here](https://github.com/lm-sys/FastChat/blob/00d9e6675bdff60be6603ffff9313b1d797d2e3e/fastchat/conversation.py#L83-L112).

### Tokenizer issues
There are some frequently asked tokenizer issues (https://github.com/lm-sys/FastChat/issues/408).
Some of them are not only related to FastChat or Vicuna weights but are also related to how you convert the base llama model.

We suggest that you use `transformers>=4.28.0` and redo the weight conversion for the base llama model.
After applying the delta, you should have a file named `special_tokens_map.json` in your converted weight folder for either v0 or v1.1.
The contents of this file should be the same as this file: https://huggingface.co/lmsys/vicuna-13b-delta-v0/blob/main/special_tokens_map.json.
If the file is not present, please copy the `special_tokens_map.json` and `tokenizer_config.json` files from https://huggingface.co/lmsys/vicuna-13b-delta-v0/tree/main to your converted weight folder. This works for both v0 and v1.1.
