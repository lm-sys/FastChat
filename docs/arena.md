# Chatbot Arena
Chatbot Arena is an LLM benchmark platform featuring anonymous, randomized battles, available at https://chat.lmsys.org.
We invite the entire community to join this benchmarking effort by contributing your votes and models.

## How to add a new model
If you want to see a specific model in the arena, you can follow the methods below.

- Method 1: Hosted by LMSYS.
  1. Contribute the code to support this model in FastChat by submitting a pull request. See [instructions](model_support.md#how-to-support-a-new-model).
  2. After the model is supported, we will try to schedule some compute resources to host the model in the arena. However, due to the limited resources we have, we may not be able to serve every model. We will select the models based on popularity, quality, diversity, and other factors.

- Method 2: Hosted by 3rd party API providers or yourself.
  1. If you have a model hosted by a 3rd party API provider or yourself, please give us an API endpoint. We prefer OpenAI-compatible APIs, so we can reuse our [code](https://github.com/lm-sys/FastChat/blob/33dca5cf12ee602455bfa9b5f4790a07829a2db7/fastchat/serve/gradio_web_server.py#L333-L358) for calling OpenAI models.
  2. You can use FastChat's OpenAI API [server](openai_api.md) to serve your model with OpenAI-compatible APIs and provide us with the endpoint.
