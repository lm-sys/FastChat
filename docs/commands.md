### Launch a service with four models on 8 V100 (16 GB) GPUs.
```
# Launch a controller
python3 -m chatserver.serve.controller

CUDA_VISIBLE_DEVICES=2 python3 -m chatserver.serve.model_worker --model /home/ubuntu/model_weights/hf-llama-7b --port 21002 --worker-address http://localhost:21002

CUDA_VISIBLE_DEVICES=3 python3 -m chatserver.serve.model_worker --model /home/ubuntu/model_weights/alpaca-7b --port 21003 --worker-address http://localhost:21003

CUDA_VISIBLE_DEVICES=4,5 python3 -m chatserver.serve.model_worker --model /home/ubuntu/model_weights/alpaca-13b --port 21004 --worker-address http://localhost:21004 --num-gpus 2

CUDA_VISIBLE_DEVICES=6,7 python3 -m chatserver.serve.model_worker --model /home/ubuntu/model_weights/bair-chat-13b --port 21006 --worker-address http://localhost:21006 --num-gpus 2

# Luanch a gradio web server.
python3 -m chatserver.serve.gradio_web_server
```

### Data cleanning
```
python3 -m chatserver.data.clean_sharegpt --in sharegpt_20230322_html.json --out sharegpt_20230322_clean.json
python3 -m chatserver.data.split_long_conversation --in sharegpt_20230322_clean.json --out sharegpt_20230322_split.json --model-name /home/ubuntu/model_weights/hf-llama-7b/

gsutil cp sharegpt_20230322_clean.json gs://model-weights/sharegpt/
gsutil cp sharegpt_20230322_split.json gs://model-weights/sharegpt/
```

### Host a gradio web server
```
sudo apt update
wget https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
pip3 install -e .
python3 -m chatserver.serve.gradio_web_server --controller http://ec2-35-89-79-20.us-west-2.compute.amazonaws.com:21001
```
