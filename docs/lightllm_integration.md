# LightLLM Integration
You can use [LightLLM](https://github.com/ModelTC/lightllm) as an optimized worker implementation in FastChat.
It offers advanced continuous batching and a much higher (~10x) throughput.
See the supported models [here](https://github.com/ModelTC/lightllm?tab=readme-ov-file#supported-model-list).

## Instructions
1. Please refer to the [Get started](https://github.com/ModelTC/lightllm?tab=readme-ov-file#get-started) to install LightLLM. Or use [Pre-built image](https://github.com/ModelTC/lightllm?tab=readme-ov-file#container)

2. When you launch a model worker, replace the normal worker (`fastchat.serve.model_worker`) with the LightLLM worker (`fastchat.serve.lightllm_worker`). All other commands such as controller, gradio web server, and OpenAI API server are kept the same. Refer to [--max_total_token_num](https://github.com/ModelTC/lightllm/blob/4a9824b6b248f4561584b8a48ae126a0c8f5b000/docs/ApiServerArgs.md?plain=1#L23) to understand how to calcuate the `--max_total_token_num` argument.
   ```
   python3 -m fastchat.serve.lightllm_worker --model-path lmsys/vicuna-7b-v1.5 --tokenizer_mode "auto" --max_total_token_num 154000
   ```

   If you what to use quantized weight and kv cache for inference, try

   ```
   python3 -m fastchat.serve.lightllm_worker --model-path lmsys/vicuna-7b-v1.5 --tokenizer_mode "auto" --max_total_token_num 154000 --mode triton_int8weight triton_int8kv
   ```
