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

Test the API server

### List Models
```bash
curl http://localhost:8000/v1/models
```

### Chat Completions
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vicuna-7b-v1.1",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Text Completions
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

### Embeddings
```bash
curl http://localhost:8000/v1/create_embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vicuna-7b-v1.1",
    "input": "Hello, can you tell me a joke"
  }'
```

## Client SDK

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

- [ ] Support more parameters like `top_p`, `presence_penalty`
- [ ] Report token usage for chat completion
- [ ] Proper error handling (e.g., model not found)
- [ ] The return value in the client SDK could be used like a dict
