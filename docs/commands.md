### Commands to launch a service with three models on 8 V100 (16 GB) GPUs.
```
# Launch a controller
python3 -m chatserver.server.controller

# Launch model workers
CUDA_VISIBLE_DEVICES=4 python3 -m chatserver.server.model_worker --model facebook/opt-350m --port 21004 --worker-address http://localhost:21004

CUDA_VISIBLE_DEVICES=5 python3 -m chatserver.server.model_worker --model facebook/opt-6.7b --port 21005 --worker-address http://localhost:21005

CUDA_VISIBLE_DEVICES=6,7 python3 -m chatserver.server.model_worker --model facebook/llama-7b --port 21006 --worker-address http://localhost:21006 --num-gpus 2

# Luanch a gradio web server.
python3 -m chatserver.server.gradio_web_server
```

