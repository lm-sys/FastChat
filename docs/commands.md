### Launch a service with three models on 8 V100 (16 GB) GPUs.
```
# Launch a controller
python3 -m chatserver.serve.controller

# Launch model workers
CUDA_VISIBLE_DEVICES=4 python3 -m chatserver.serve.model_worker --model facebook/opt-350m --port 21004 --worker-address http://localhost:21004

CUDA_VISIBLE_DEVICES=5 python3 -m chatserver.serve.model_worker --model facebook/opt-6.7b --port 21005 --worker-address http://localhost:21005

CUDA_VISIBLE_DEVICES=6,7 python3 -m chatserver.serve.model_worker --model facebook/llama-7b --port 21006 --worker-address http://localhost:21006 --num-gpus 2

# Luanch a gradio web server.
python3 -m chatserver.serve.gradio_web_server
```


### Host a gradio web server
```
sudo apt update
wget https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
pip3 install -e .
python3 -m chatserver.serve.gradio_web_server --controller http://ec2-35-89-79-20.us-west-2.compute.amazonaws.com:21001
```
