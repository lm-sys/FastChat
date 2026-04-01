# GGUF/GGML Integration

You can use [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) as an optimized worker implementation in FastChat to load quantized GGUF/GGML models.

This is useful for running large language models efficiently on CPU or with partial GPU offloading, using quantized model formats that significantly reduce memory requirements.

## Instructions

1. Install llama-cpp-python.

   ```
   pip install "llama-cpp-python>=0.2.0"
   ```

   For GPU acceleration (CUDA):

   ```
   CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
   ```

   For GPU acceleration (Metal / Apple Silicon):

   ```
   CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
   ```

   Or install via FastChat's optional dependency:

   ```
   pip install "fschat[gguf]"
   ```

2. When you launch a model worker, replace the normal worker (`fastchat.serve.model_worker`) with the GGUF worker (`fastchat.serve.gguf_worker`). Remember to launch a controller first ([instructions](../README.md)).

   ```
   python3 -m fastchat.serve.gguf_worker --model-path /path/to/model.gguf
   ```

3. Optional arguments:

   - `--n-gpu-layers`: Number of layers to offload to GPU. Use `-1` for all layers. Default: `0` (CPU only).
   - `--n-ctx`: Context window size. Default: `2048`.
   - `--n-batch`: Batch size for prompt processing. Default: `512`.
   - `--model-names`: Comma-separated display names for the model.
   - `--conv-template`: Conversation prompt template name.

## Example

```
# Launch controller
python3 -m fastchat.serve.controller

# Launch GGUF worker with GPU offloading
python3 -m fastchat.serve.gguf_worker \
    --model-path ./models/llama-2-7b-chat.Q4_K_M.gguf \
    --model-names llama-2-7b-chat \
    --conv-template llama-2 \
    --n-gpu-layers -1 \
    --n-ctx 4096

# Launch API server
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
```
