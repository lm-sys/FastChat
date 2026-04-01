# fastllm Integration

You can use [fastllm](https://github.com/ztxz16/fastllm) as an optimized worker implementation in FastChat.

fastllm is a high-performance LLM inference engine that supports both CPU and GPU, with especially strong CPU performance through custom acceleration kernels. It supports multiple model architectures including ChatGLM, LLaMA, MOSS, and more.

## Instructions

1. Install fastllm following the [official guide](https://github.com/ztxz16/fastllm#build).

   ```bash
   git clone https://github.com/ztxz16/fastllm
   cd fastllm
   mkdir build && cd build
   cmake .. -DUSE_CUDA=ON  # or -DUSE_CUDA=OFF for CPU-only
   make -j
   cd ../
   cd pyfastllm && python setup.py install
   ```

2. When you launch a model worker, replace the normal worker (`fastchat.serve.model_worker`) with the fastllm worker (`fastchat.serve.fastllm_worker`). Remember to launch a controller first ([instructions](../README.md)).

   ```bash
   python3 -m fastchat.serve.fastllm_worker --model-path chatglm2-6b
   ```

3. You can specify additional options:

   ```bash
   python3 -m fastchat.serve.fastllm_worker \
       --model-path chatglm2-6b \
       --dtype int8 \
       --threads 8 \
       --context-length 4096
   ```

## Supported Options

| Option              | Default     | Description                                         |
|---------------------|-------------|-----------------------------------------------------|
| `--model-path`      | (required)  | Path to model (HuggingFace format or `.flm` file)   |
| `--dtype`           | `float16`   | Weight data type: `float16`, `float32`, `int8`, `int4` |
| `--threads`         | `4`         | Number of CPU threads                               |
| `--context-length`  | `2048`      | Maximum context length                              |
| `--conv-template`   | auto-detect | Conversation prompt template name                   |
