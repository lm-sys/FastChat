set -x

curl http://localhost:8000/v1/models

echo

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vicuna-7b-v1.1",
    "messages": [{"role": "user", "content": "Hello! What is your name?"}]
  }'

echo

curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vicuna-7b-v1.1",
    "prompt": "Once upon a time",
    "max_tokens": 41,
    "temperature": 0.5
  }'

echo

curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vicuna-7b-v1.1",
    "input": "Hello world!"
  }'
