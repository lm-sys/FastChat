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
