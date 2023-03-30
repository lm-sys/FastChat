### Local GPU cluster (node-01)
```
python3 -m fastchat.serve.controller --host 0.0.0.0 --port 10002

CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.model_worker --model-path ~/model_weights/vicuna-13b/ --controller http://localhost:10002 --port 31000 --worker http://localhost:31000
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.model_worker --model-path ~/model_weights/vicuna-13b/ --controller http://localhost:10002 --port 31001 --worker http://localhost:31001
CUDA_VISIBLE_DEVICES=2 python3 -m fastchat.serve.model_worker --model-path ~/model_weights/bair-chat-13b/ --controller http://localhost:10002 --port 31002 --worker http://localhost:31002
CUDA_VISIBLE_DEVICES=3 python3 -m fastchat.serve.model_worker --model-path ~/model_weights/alpaca-chat-13b/ --controller http://localhost:10002 --port 31003 --worker http://localhost:31003

python3 -m fastchat.serve.test_message --model vicuna-13b --controller http://localhost:10002
```

### Web server
```
python3 -m fastchat.serve.controller --host 0.0.0.0 --port 21001

python3 -m fastchat.serve.register_worker --controller http://localhost:21001 --worker-name https://

python3 -m fastchat.serve.test_message --model vicuna-13b --controller http://localhost:21001

python3 -m fastchat.serve.gradio_web_server --controller http://localhost:21001
```

### Local GPU cluster (node-02)
```
CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.model_worker --model-path ~/model_weights/vicuna-13b/ --controller http://node-01:10002 --host 0.0.0.0 --port 31000 --worker http://$(hostname):31000
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.model_worker --model-path ~/model_weights/vicuna-13b/ --controller http://node-01:10002 --host 0.0.0.0 --port 31001 --worker http://$(hostname):31001
CUDA_VISIBLE_DEVICES=2 python3 -m fastchat.serve.model_worker --model-path ~/model_weights/vicuna-13b/ --controller http://node-01:10002 --host 0.0.0.0 --port 31002 --worker http://$(hostname):31002
CUDA_VISIBLE_DEVICES=3 python3 -m fastchat.serve.model_worker --model-path ~/model_weights/vicuna-13b/ --controller http://node-01:10002 --host 0.0.0.0 --port 31003 --worker http://$(hostname):31003
```
