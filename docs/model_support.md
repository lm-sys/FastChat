# Model Support

## Supported models
- Vicuna, Alpaca, LLaMA, Koala
   - example: `python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.3`
- [BlinkDL/RWKV-4-Raven](https://huggingface.co/BlinkDL/rwkv-4-raven)
   - example: `python3 -m fastchat.serve.cli --model-path ~/model_weights/RWKV-4-Raven-7B-v11x-Eng99%-Other1%-20230429-ctx8192.pth`
- [camel-ai/CAMEL-13B-Combined-Data](https://huggingface.co/camel-ai/CAMEL-13B-Combined-Data)
- [databricks/dolly-v2-12b](https://huggingface.co/databricks/dolly-v2-12b)
- [FreedomIntelligence/phoenix-inst-chat-7b](https://huggingface.co/FreedomIntelligence/phoenix-inst-chat-7b)
- [h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b](https://huggingface.co/h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b)
- [lcw99/polyglot-ko-12.8b-chang-instruct-chat](https://huggingface.co/lcw99/polyglot-ko-12.8b-chang-instruct-chat)
- [lmsys/fastchat-t5-3b-v1.0](https://huggingface.co/lmsys/fastchat-t5)
- [mosaicml/mpt-7b-chat](https://huggingface.co/mosaicml/mpt-7b-chat)
  - example: `python3 -m fastchat.serve.cli --model-path mosaicml/mpt-7b-chat`
- [Neutralzz/BiLLa-7B-SFT](https://huggingface.co/Neutralzz/BiLLa-7B-SFT)
- [nomic-ai/gpt4all-13b-snoozy](https://huggingface.co/nomic-ai/gpt4all-13b-snoozy)
- [openaccess-ai-collective/manticore-13b-chat-pyg](https://huggingface.co/openaccess-ai-collective/manticore-13b-chat-pyg)
- [OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5](https://huggingface.co/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5)
- [project-baize/baize-v2-7b](https://huggingface.co/project-baize/baize-v2-7b)
- [StabilityAI/stablelm-tuned-alpha-7b](https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b)
- [THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)
- [THUDM/chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b)
- [tiiuae/falcon-40b](https://huggingface.co/tiiuae/falcon-40b)
- [timdettmers/guanaco-33b-merged](https://huggingface.co/timdettmers/guanaco-33b-merged)
- [togethercomputer/RedPajama-INCITE-7B-Chat](https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Chat)
- [WizardLM/WizardLM-13B-V1.0](https://huggingface.co/WizardLM/WizardLM-13B-V1.0)
- [baichuan-inc/baichuan-7B](https://huggingface.co/baichuan-inc/baichuan-7B)
- Any [Peft](https://github.com/huggingface/peft) adapter trained ontop of a model above.  To activate, must have `peft` in the model path.

## How to support a new model

To support a new model in FastChat, you need to correctly handle its prompt template and model loading.
The goal is to make the following command run with the correct prompts.
```
python3 -m fastchat.serve.cli --model [YOUR_MODEL_PATH]
```

You can run this example command to learn the code logic.
```
python3 -m fastchat.serve.cli --model lmsys/vicuna-7b-v1.3
```

You can add `--debug` to see the actual prompt sent to the model.

### Steps
FastChat uses the `Conversation` class to handle prompt templates and `BaseModelAdapter` class to handle model loading.

1. Implement a conversation template for the new model at [fastchat/conversation.py](https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py). You can follow existing examples and use `register_conv_template` to add a new one.
2. Implement a model adapter for the new model at [fastchat/model/model_adapter.py](https://github.com/lm-sys/FastChat/blob/main/fastchat/model/model_adapter.py). You can follow existing examples and use `register_model_adapter` to add a new one.
3. (Optional) add the model name to the "Supported models" [section](#supported-models) above and add more information in [fastchat/model/model_registry.py](https://github.com/lm-sys/FastChat/blob/main/fastchat/model/model_registry.py).

After these steps, the new model should be compatible with most FastChat features, such as CLI, web UI, model worker, and OpenAI-compatible API server. Please do some testing with these features as well.
