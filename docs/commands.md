### Launch a service with four models on 8 V100 (16 GB) GPUs.
```
# Launch a controller
python3 -m fastchat.serve.controller

CUDA_VISIBLE_DEVICES=2 python3 -m fastchat.serve.model_worker --model /home/ubuntu/model_weights/hf-llama-7b --port 21002 --worker-address http://localhost:21002

CUDA_VISIBLE_DEVICES=3 python3 -m fastchat.serve.model_worker --model /home/ubuntu/model_weights/alpaca-7b --port 21003 --worker-address http://localhost:21003

CUDA_VISIBLE_DEVICES=4,5 python3 -m fastchat.serve.model_worker --model /home/ubuntu/model_weights/alpaca-13b --port 21004 --worker-address http://localhost:21004 --num-gpus 2

CUDA_VISIBLE_DEVICES=6,7 python3 -m fastchat.serve.model_worker --model /home/ubuntu/model_weights/bair-chat-13b --port 21006 --worker-address http://localhost:21006 --num-gpus 2

# Luanch a gradio web server.
python3 -m fastchat.serve.gradio_web_server
```

### Data cleanning
```
python3 -m fastchat.data.clean_sharegpt --in sharegpt_20230322_html.json --out sharegpt_20230322_clean.json
python3 -m fastchat.data.optional_clean --in sharegpt_20230322_clean.json --out sharegpt_20230322_clean_lang.json --skip-lang
python3 -m fastchat.data.split_long_conversation --in sharegpt_20230322_clean_lang.json --out sharegpt_20230322_clean_lang_split.json --model-name /home/ubuntu/model_weights/hf-llama-7b/

gsutil cp sharegpt_20230322_clean.json gs://model-weights/sharegpt/
gsutil cp sharegpt_20230322_split.json gs://model-weights/sharegpt/
```

### Local Cluster

#### Local GPU cluster (node-01)
```
python3 -m fastchat.serve.controller --host 0.0.0.0 --port 10002

CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.model_worker --model-path ~/model_weights/alpaca-7b/ --controller http://localhost:10002 --port 31000 --worker http://localhost:31000
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.model_worker --model-path ~/model_weights/alpaca-13b/ --controller http://localhost:10002 --port 31001 --worker http://localhost:31001
CUDA_VISIBLE_DEVICES=2 python3 -m fastchat.serve.model_worker --model-path ~/model_weights/bair-chat-7b/ --controller http://localhost:10002 --port 31002 --worker http://localhost:31002
CUDA_VISIBLE_DEVICES=3 python3 -m fastchat.serve.model_worker --model-path ~/model_weights/bair-chat-13b/ --controller http://localhost:10002 --port 31003 --worker http://localhost:31003

python3 -m fastchat.serve.test_message --model alpaca-7b --controller http://localhost:10002
```

#### Web server
```
python3 -m fastchat.serve.controller --host 0.0.0.0 --port 21001

python3 -m fastchat.serve.register_worker --controller http://localhost:21001 --worker-name https://
python3 -m fastchat.serve.test_message --model alpaca-7b --controller http://localhost:21001

python3 -m fastchat.serve.gradio_web_server --controller http://localhost:21001
```

#### Local GPU cluster (node-02)
```
CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.model_worker --model-path ~/model_weights/vicuna-13b/ --controller http://node-01:10002 --host 0.0.0.0 --port 31000 --worker http://$(hostname):31000
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.model_worker --model-path ~/model_weights/vicuna-13b/ --controller http://node-01:10002 --host 0.0.0.0 --port 31001 --worker http://$(hostname):31001
CUDA_VISIBLE_DEVICES=2 python3 -m fastchat.serve.model_worker --model-path ~/model_weights/vicuna-13b/ --controller http://node-01:10002 --host 0.0.0.0 --port 31002 --worker http://$(hostname):31002
CUDA_VISIBLE_DEVICES=3 python3 -m fastchat.serve.model_worker --model-path ~/model_weights/vicuna-13b/ --controller http://node-01:10002 --host 0.0.0.0 --port 31003 --worker http://$(hostname):31003
```

### Host a gradio web server
```
sudo apt update
wget https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
pip3 install -e .
python3 -m fastchat.serve.gradio_web_server --controller http://ec2-35-89-79-20.us-west-2.compute.amazonaws.com:21001
```
