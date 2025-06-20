# vLLM Integration
You can use [LMDeploy](https://lmdeploy.readthedocs.io/en/latest/) as an optimized worker implementation in FastChat.
It offers advanced continuous batching and a much higher (~10x) throughput.
See the supported models [here](https://lmdeploy.readthedocs.io/en/latest/supported_models/supported_models.html).

## Instructions
1. Install LMDeploy.
    ```
    pip install lmdeploy
    ```

2. When you launch a model worker, replace the normal worker (`fastchat.serve.model_worker`) with the LMDeploy worker (`fastchat.serve.lmdeploy_worker`). All other commands such as controller, gradio web server, and OpenAI API server are kept the same.
   ```
   python3 -m fastchat.serve.lmdeploy_worker --model-path lmsys/vicuna-7b-v1.5
   ```

   If you use an AWQ quantized model, try
   '''
   python3 -m fastchat.serve.lmdeploy_worker --model-path TheBloke/LLaMA2-13B-Tiefighter-AWQ --model-format awq
   '''
