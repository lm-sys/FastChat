# vLLM Integration
You can use [vLLM](https://vllm.ai/) as an optimized worker implementation in FastChat.
It offers advanced continuous batching and a much higher (~10x) throughput.
See the supported models [here](https://vllm.readthedocs.io/en/latest/models/supported_models.html).

## Instructions
1. Install vLLM.
    ```
    pip install vllm
    ```

2. When you launch a model worker, replace the normal worker (`fastchat.serve.model_worker`) with the vLLM worker (`fastchat.serve.vllm_worker`). All other commands such as controller, gradio web server, and OpenAI API server are kept the same.
   ```
   python3 -m fastchat.serve.vllm_worker --model-path lmsys/vicuna-7b-v1.5
   ```

   If you see tokenizer errors, try
   ```
   python3 -m fastchat.serve.vllm_worker --model-path lmsys/vicuna-7b-v1.5 --tokenizer hf-internal-testing/llama-tokenizer
   ```

   If you use an AWQ quantized model, try
   '''
   python3 -m fastchat.serve.vllm_worker --model-path TheBloke/vicuna-7B-v1.5-AWQ --quantization awq
   '''

## Add vllm_worker support for lora_modules

### usage

1. start

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m fastchat.serve.vllm_worker \
    --model-path /data/models/Qwen/Qwen2-72B-Instruct \
    --tokenizer /data/models/Qwen/Qwen2-72B-Instruct  \
    --enable-lora \
    --lora-modules m1=/data/modules/lora/adapter/m1 m2=/data/modules/lora/adapter/m2 m3=/data/modules/lora/adapter/m3 \
    --model-names qwen2-72b-instruct,m1,m2,m3\
    --controller http://localhost:21001 \
    --host 0.0.0.0 \
    --num-gpus 8 \
    --port 31034 \
    --limit-worker-concurrency 100 \
    --worker-address http://localhost:31034
```

1. post

- example1

```bash
curl --location --request POST 'http://fastchat_address:port/v1/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer sk-xxx' \
--data-raw '{
    "model": "m1",
    "stream": false,
    "temperature": 0.7,
    "top_p": 0.1,
    "max_tokens": 4096,
    "messages": [
      {
        "role": "user",
        "content": "Hi?"
      }
    ]
  }'
```

- example2

```bash
curl --location --request POST 'http://fastchat_address:port/v1/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer sk-xxx' \
--data-raw '{
    "model": "qwen2-72b-instruct",
    "stream": false,
    "temperature": 0.7,
    "top_p": 0.1,
    "max_tokens": 4096,
    "messages": [
      {
        "role": "user",
        "content": "Hi?"
      }
    ]
  }'
```
