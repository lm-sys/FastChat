# Chatbot Arena
Chatbot Arena is an LLM benchmark platform featuring anonymous, randomized battles, available at https://chat.lmsys.org.
We invite the entire community to join this benchmarking effort by contributing your votes and models.

## How to add a new model
If you want to see a specific model in the arena, you can follow the methods below.

### Method 1: Hosted by 3rd party API providers or yourself
If you have a model hosted by a 3rd party API provider or yourself, please give us the access to an API endpoint.
  - We prefer OpenAI-compatible APIs, so we can reuse our [code](https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/api_provider.py) for calling OpenAI models.
  - If you have your own API protocol, please follow the [instructions](model_support.md) to add them. Contribute your code by sending a pull request.

### Method 2: Hosted by LMSYS
1. Contribute the code to support this model in FastChat by submitting a pull request. See [instructions](model_support.md).
2. After the model is supported, we will try to schedule some compute resources to host the model in the arena. However, due to the limited resources we have, we may not be able to serve every model. We will select the models based on popularity, quality, diversity, and other factors.
