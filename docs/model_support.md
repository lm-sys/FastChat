# Model Support
This document describes how to support a new model in FastChat.

## Content
- [Local Models](#local-models)
- [API-Based Models](#api-based-models)

## Local Models
To support a new local model in FastChat, you need to correctly handle its prompt template and model loading.
The goal is to make the following command run with the correct prompts.

```
python3 -m fastchat.serve.cli --model [YOUR_MODEL_PATH]
```

You can run this example command to learn the code logic.

```
python3 -m fastchat.serve.cli --model lmsys/vicuna-7b-v1.5
```

You can add `--debug` to see the actual prompt sent to the model.

### Steps

FastChat uses the `Conversation` class to handle prompt templates and `BaseModelAdapter` class to handle model loading.

1. Implement a conversation template for the new model at [fastchat/conversation.py](https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py). You can follow existing examples and use `register_conv_template` to add a new one. Please also add a link to the official reference code if possible.
2. Implement a model adapter for the new model at [fastchat/model/model_adapter.py](https://github.com/lm-sys/FastChat/blob/main/fastchat/model/model_adapter.py). You can follow existing examples and use `register_model_adapter` to add a new one.
3. (Optional) add the model name to the "Supported models" [section](#supported-models) above and add more information in [fastchat/model/model_registry.py](https://github.com/lm-sys/FastChat/blob/main/fastchat/model/model_registry.py).

After these steps, the new model should be compatible with most FastChat features, such as CLI, web UI, model worker, and OpenAI-compatible API server. Please do some testing with these features as well.

### Supported models

- [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
  - example: `python3 -m fastchat.serve.cli --model-path meta-llama/Llama-2-7b-chat-hf`
- Vicuna, Alpaca, LLaMA, Koala
  - example: `python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5`
- [Alibaba-NLP/gte-large-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5)
- [allenai/tulu-2-dpo-7b](https://huggingface.co/allenai/tulu-2-dpo-7b)
- [BAAI/AquilaChat-7B](https://huggingface.co/BAAI/AquilaChat-7B)
- [BAAI/AquilaChat2-7B](https://huggingface.co/BAAI/AquilaChat2-7B)
- [BAAI/AquilaChat2-34B](https://huggingface.co/BAAI/AquilaChat2-34B)
- [BAAI/bge-large-en](https://huggingface.co/BAAI/bge-large-en#using-huggingface-transformers)
- [argilla/notus-7b-v1](https://huggingface.co/argilla/notus-7b-v1)
- [baichuan-inc/baichuan-7B](https://huggingface.co/baichuan-inc/baichuan-7B)
- [BlinkDL/RWKV-4-Raven](https://huggingface.co/BlinkDL/rwkv-4-raven)
  - example: `python3 -m fastchat.serve.cli --model-path ~/model_weights/RWKV-4-Raven-7B-v11x-Eng99%-Other1%-20230429-ctx8192.pth`
- [bofenghuang/vigogne-2-7b-instruct](https://huggingface.co/bofenghuang/vigogne-2-7b-instruct)
- [bofenghuang/vigogne-2-7b-chat](https://huggingface.co/bofenghuang/vigogne-2-7b-chat)
- [camel-ai/CAMEL-13B-Combined-Data](https://huggingface.co/camel-ai/CAMEL-13B-Combined-Data)
- [codellama/CodeLlama-7b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf)
- [databricks/dolly-v2-12b](https://huggingface.co/databricks/dolly-v2-12b)
- [deepseek-ai/deepseek-llm-67b-chat](https://huggingface.co/deepseek-ai/deepseek-llm-67b-chat)
- [deepseek-ai/deepseek-coder-33b-instruct](https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct)
- [FlagAlpha/Llama2-Chinese-13b-Chat](https://huggingface.co/FlagAlpha/Llama2-Chinese-13b-Chat)
- [FreedomIntelligence/phoenix-inst-chat-7b](https://huggingface.co/FreedomIntelligence/phoenix-inst-chat-7b)
- [FreedomIntelligence/ReaLM-7b-v1](https://huggingface.co/FreedomIntelligence/Realm-7b)
- [h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b](https://huggingface.co/h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b)
- [HuggingFaceH4/starchat-beta](https://huggingface.co/HuggingFaceH4/starchat-beta)
- [HuggingFaceH4/zephyr-7b-alpha](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha)
- [internlm/internlm-chat-7b](https://huggingface.co/internlm/internlm-chat-7b)
- [cllm/consistency-llm-7b-codesearchnet/consistency-llm-7b-gsm8k/consistency-llm-7b-sharegpt48k/consistency-llm-7b-spider](https://huggingface.co/cllm)
- [IEITYuan/Yuan2-2B/51B/102B-hf](https://huggingface.co/IEITYuan)
- [lcw99/polyglot-ko-12.8b-chang-instruct-chat](https://huggingface.co/lcw99/polyglot-ko-12.8b-chang-instruct-chat)
- [lmsys/fastchat-t5-3b-v1.0](https://huggingface.co/lmsys/fastchat-t5)
- [meta-math/MetaMath-7B-V1.0](https://huggingface.co/meta-math/MetaMath-7B-V1.0)
- [Microsoft/Orca-2-7b](https://huggingface.co/microsoft/Orca-2-7b)
- [moka-ai/m3e-large](https://huggingface.co/moka-ai/m3e-large)
- [mosaicml/mpt-7b-chat](https://huggingface.co/mosaicml/mpt-7b-chat)
  - example: `python3 -m fastchat.serve.cli --model-path mosaicml/mpt-7b-chat`
- [Neutralzz/BiLLa-7B-SFT](https://huggingface.co/Neutralzz/BiLLa-7B-SFT)
- [nomic-ai/gpt4all-13b-snoozy](https://huggingface.co/nomic-ai/gpt4all-13b-snoozy)
- [NousResearch/Nous-Hermes-13b](https://huggingface.co/NousResearch/Nous-Hermes-13b)
- [openaccess-ai-collective/manticore-13b-chat-pyg](https://huggingface.co/openaccess-ai-collective/manticore-13b-chat-pyg)
- [OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5](https://huggingface.co/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5)
- [openchat/openchat_3.5](https://huggingface.co/openchat/openchat_3.5)
- [Open-Orca/Mistral-7B-OpenOrca](https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca)
- [OpenLemur/lemur-70b-chat-v1](https://huggingface.co/OpenLemur/lemur-70b-chat-v1)
- [Phind/Phind-CodeLlama-34B-v2](https://huggingface.co/Phind/Phind-CodeLlama-34B-v2)
- [project-baize/baize-v2-7b](https://huggingface.co/project-baize/baize-v2-7b)
- [Qwen/Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat)
- [rishiraj/CatPPT](https://huggingface.co/rishiraj/CatPPT)
- [Salesforce/codet5p-6b](https://huggingface.co/Salesforce/codet5p-6b)
- [shibing624/text2vec-base-multilingual](https://huggingface.co/shibing624/text2vec-base-multilingual)
- [StabilityAI/stablelm-tuned-alpha-7b](https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b)
- [tenyx/TenyxChat-7B-v1](https://huggingface.co/tenyx/TenyxChat-7B-v1)
- [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)
- [THUDM/chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b)
- [tiiuae/falcon-40b](https://huggingface.co/tiiuae/falcon-40b)
- [tiiuae/falcon-180B-chat](https://huggingface.co/tiiuae/falcon-180B-chat)
- [timdettmers/guanaco-33b-merged](https://huggingface.co/timdettmers/guanaco-33b-merged)
- [togethercomputer/RedPajama-INCITE-7B-Chat](https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Chat)
- [VMware/open-llama-7b-v2-open-instruct](https://huggingface.co/VMware/open-llama-7b-v2-open-instruct)
- [WizardLM/WizardLM-13B-V1.0](https://huggingface.co/WizardLM/WizardLM-13B-V1.0)
- [WizardLM/WizardCoder-15B-V1.0](https://huggingface.co/WizardLM/WizardCoder-15B-V1.0)
- [Xwin-LM/Xwin-LM-7B-V0.1](https://huggingface.co/Xwin-LM/Xwin-LM-70B-V0.1)
- Any [EleutherAI](https://huggingface.co/EleutherAI) pythia model such as [pythia-6.9b](https://huggingface.co/EleutherAI/pythia-6.9b)
- Any [Peft](https://github.com/huggingface/peft) adapter trained on top of a
  model above.  To activate, must have `peft` in the model path.  Note: If
  loading multiple peft models, you can have them share the base model weights by
  setting the environment variable `PEFT_SHARE_BASE_WEIGHTS=true` in any model
  worker.


## API-Based Models
To support an API-based model, consider learning from the existing OpenAI example.
If the model is compatible with OpenAI APIs, then a configuration file is all that's needed without any additional code.
For custom protocols, implementation of a streaming generator in [fastchat/serve/api_provider.py](https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/api_provider.py) is required, following the provided examples. Currently, FastChat is compatible with OpenAI, Anthropic, Google Vertex AI, Mistral, Nvidia NGC, YandexGPT and Reka.

### Steps to Launch a WebUI with an API Model
1. Specify the endpoint information in a JSON configuration file. For instance, create a file named `api_endpoints.json`:
```json
{
  "gpt-3.5-turbo": {
    "model_name": "gpt-3.5-turbo",
    "api_type": "openai",
    "api_base": "https://api.openai.com/v1",
    "api_key": "sk-******",
    "anony_only": false,
    "recommended_config": {
      "temperature": 0.7,
      "top_p": 1.0
    },
    "text-arena": true,
    "vision-arena": false,
  }
}
```
  - "api_type" can be one of the following: openai, anthropic, gemini, mistral, yandexgpt or reka. For custom APIs, add a new type and implement it accordingly.
  - "anony_only" indicates whether to display this model in anonymous mode only.
  - "recommended_config" indicates the recommended generation parameters for temperature and top_p.
  - "text-arena" indicates whether the model should be displayed in the Text Arena.
  - "vision-arena" indicates whether the model should be displayed in the Vision Arena.

2. Launch the Gradio web server with the argument `--register api_endpoints.json`:
```
python3 -m fastchat.serve.gradio_web_server --controller "" --share --register api_endpoints.json
```

Now, you can open a browser and interact with the model.
