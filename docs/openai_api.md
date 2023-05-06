# Openai API
<!-- (Experimental. We will keep improving the API and SDK.) -->

## Chat Completion

Reference: https://platform.openai.com/docs/api-reference/chat/create

Some features/compatibilities to be implemented:

- [ ] streaming
- [ ] support of some parameters like `top_p`, `presence_penalty`
- [ ] proper error handling (e.g. model not found)
- [ ] the return value in the client SDK could be used like a dict

## Create Embedding

Reference: https://platform.openai.com/docs/api-reference/embeddings

## Text Completion

Reference: https://platform.openai.com/docs/api-reference/completions/create


**RESTful API Server**

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
export FASTCHAT_CONTROLLER_URL=http://localhost:21001
python3 -m fastchat.serve.api --host localhost --port 8000
```

Test the API server

```bash
# chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vicuna-7b-v1.1",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```
```bash
# create embedding
curl http://localhost:8000/v1/create_embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vicuna-7b-v1.1",
    "input": "Hello, can you tell me a joke"
  }'
```
```bash
# text completion
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vicuna-7b-v1.1",
    "prompt": "Once upon a time",
    "max_tokens": 41,
    "temperature": 0.5
  }'
```

**Client SDK**

Assuming environment variable `FASTCHAT_BASEURL` is set to the API server URL (e.g., `http://localhost:8000`), you can use the following code to send a request to the API server:

```python
import os
from fastchat import client

client.set_baseurl(os.getenv("FASTCHAT_BASEURL"))

completion = client.ChatCompletion.create(
  model="vicuna-7b-v1.1",
  messages=[
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)
```

**Machine Learning with Embeddings**

See [fastchat/playground/test_embedding/test_sentence_similarity.py](../playground/test_embedding/test_sentence_similarity.py)

Feel free to use `create_embedding` to 
- build your own classifier, see [fastchat/playground/test_embedding/test_classification.py](../playground/test_embedding/test_classification.py)
- evaluate texts' similarity, see [fastchat/playground/test_embedding/test_sentence_similarity.py](../playground/test_embedding/test_sentence_similarity.py)
- search relative texts, see [fastchat/playground/test_embedding/test_semantic_search.py](../playground/test_embedding/test_semantic_search.py)

To run the tests, you need to download the data [here](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews), and openai api key is required to make comparation.

Run with:
~~~bash
python3 playground/test_embedding/test_classification.py
~~~
and you will train a classifier based on `vicuna-7b`, `text-similarity-ada-001` and `text-embedding-ada-002`
