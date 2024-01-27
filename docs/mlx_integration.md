# Apple MLX Integration

You can use [Apple MLX](https://github.com/ml-explore/mlx) as an optimized worker implementation in FastChat.

It runs models efficiently on Apple Silicon

See the supported models [here](https://github.com/ml-explore/mlx-examples/tree/main/llms#supported-models).

Note that for Apple Silicon Macs with less memory, smaller models (or quantized models) are recommended.

## Instructions

1. Install MLX.

   ```
   pip install mlx-lm
   ```

2. When you launch a model worker, replace the normal worker (`fastchat.serve.model_worker`) with the MLX worker (`fastchat.serve.mlx_worker`).

   ```
   python3 -m fastchat.serve.mlx_worker --model-path microsoft/phi-2
   ```
