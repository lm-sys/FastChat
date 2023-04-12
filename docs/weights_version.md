## Vicuna-7B/13B

| Weights Version | v1.1 | v0 |
| ---- | ---- | ---- |
| Link      | [7B](https://huggingface.co/lmsys/vicuna-7b-delta-v1.1), [13B](https://huggingface.co/lmsys/vicuna-13b-delta-v1.1) | [7B](https://huggingface.co/lmsys/vicuna-7b-delta-v0), [13B](https://huggingface.co/lmsys/vicuna-13b-delta-v0) |
| Separator | `</s>` | `###` |
| FastChat PyPI package compatibility | >= v0.2.0 |<= v0.1.10 |

Major updates of Weights v1.1
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

See the full prompt template [here]().

### Example prompt (Weight v0)
```
A chat between a human and an assistant.

### Human: Hello!
### Assistant: Hello!
### Human: How are you?
### Assistant: I am good.
```

See the full prompt template [here]().
