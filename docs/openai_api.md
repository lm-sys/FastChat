# OpenAI-Compatible RESTful APIs & SDK

FastChat provides OpenAI-Compatible RESTful APIs for the its supported models (e.g. Vicuna).
The following OpenAI APIs are supported:
- Chat Completions. (Reference: https://platform.openai.com/docs/api-reference/chat)
- Completions. (Reference: https://platform.openai.com/docs/api-reference/completions)
- Embeddings. (Reference: https://platform.openai.com/docs/api-reference/embeddings)

## RESTful API Server
First, launch the controller

```bash
python3 -m fastchat.serve.controller
```

Then, launch the model worker(s)

```bash
python3 -m fastchat.serve.model_worker --model-name 'vicuna-7b-v1.1' --model-path /path/to/vicuna/weights
```

Finally, launch the RESTful API server

```bash
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
```

Now, let us test the API server...

### OpenAI Official SDK
The final goal of `openai_api_server.py` is to implement a fully OpenAI-Compatible API server, so the models can be used directly with [openai-python](https://github.com/openai/openai-python) library.

First, install openai-python:
```bash
pip install --upgrade openai
```

Then, interact with model vicuna:
```python
import openai
openai.api_key = "EMPTY" # Not support yet
openai.api_base = "http://localhost:8000/v1"

# create a completion
completion = openai.Completion.create(model="vicuna-7b-v1.1", prompt="Hello world", max_tokens=64)
# print the completion
print(completion.choices[0].text)

# create a chat completion
completion = openai.ChatCompletion.create(
  model="vicuna-7b-v1.1",
  messages=[{"role": "user", "content": "Hello world!"}]
)
# print the completion
print(completion.choices[0].message.content)
```

### cURL
cURL is another good tool for observing the output of the api.

List Models:
```bash
curl http://localhost:8000/v1/models
```

Chat Completions:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vicuna-7b-v1.1",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Text Completions:
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vicuna-7b-v1.1",
    "prompt": "Once upon a time",
    "max_tokens": 41,
    "temperature": 0.5
  }'
```

Embeddings:
```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vicuna-7b-v1.1",
    "input": "Hello, can you tell me a joke"
  }'
```

### FastChat Client SDK
FastChat also includes its own client SDK for the API.

Assuming environment variable `FASTCHAT_BASEURL` is set to the API server URL (e.g., `http://localhost:8000`), you can use the following code to send a request to the API server:

```python
import os
from fastchat.client import openai_api_client as client

client.set_baseurl(os.getenv("FASTCHAT_BASEURL", "http://localhost:8000"))

completion = client.ChatCompletion.create(
  model="vicuna-7b-v1.1",
  messages=[
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message.content)
```

### Streaming
See [test_openai_client.py](../tests/test_openai_client.py).

## Machine Learning with Embeddings
You can use `create_embedding` to 
- Build your own classifier, see [fastchat/playground/test_embedding/test_classification.py](../playground/test_embedding/test_classification.py)
- Evaluate text similarity, see [fastchat/playground/test_embedding/test_sentence_similarity.py](../playground/test_embedding/test_sentence_similarity.py)
- Search relative texts, see [fastchat/playground/test_embedding/test_semantic_search.py](../playground/test_embedding/test_semantic_search.py)

To these tests, you need to download the data [here](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews). You also need an OpenAI API key for comparison.

Run with:
```bash
cd playground/test_embedding
python3 test_classification.py
```
The script will train classifiers based on `vicuna-7b`, `text-similarity-ada-001` and `text-embedding-ada-002` and report the accuracy of each classifier.

## Todos
Some features to be implemented:

- [ ] Support more parameters like `logprobs`, `logit_bias`, `user`, `presence_penalty` and `frequency_penalty`
- [ ] The return value in the client SDK could be used like a dict
- [ ] Model details (permissions, owner and create time)
- [ ] Edits API
- [ ] Authentication and API key
- [ ] Rate Limitation Settings
- [x] Parameter `top_p` support
- [x] Report token usage for chat completion
- [x] Proper error handling (e.g., model not found)