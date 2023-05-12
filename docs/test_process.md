### Test CLI Inference

```
CUDA_VISIBLE_DEVICES=0,1 python3 -m fastchat.serve.cli --model ~/model_weights/koala-13b --num-gpus 2 --debug
CUDA_VISIBLE_DEVICES=2,3 python3 -m fastchat.serve.cli --model ~/model_weights/alpaca-13b --num-gpus 2 --debug
CUDA_VISIBLE_DEVICES=4,5 python3 -m fastchat.serve.cli --model ~/model_weights/vicuna-13b --num-gpus 2 --debug
CUDA_VISIBLE_DEVICES=6,7 python3 -m fastchat.serve.cli --model OpenAssistant/oasst-sft-1-pythia-12b --num-gpus 2 --debug

CUDA_VISIBLE_DEVICES=0,1 python3 -m fastchat.serve.cli --model StabilityAI/stablelm-tuned-alpha-7b --num-gpus 2 --debug
CUDA_VISIBLE_DEVICES=2,3 python3 -m fastchat.serve.cli --model databricks/dolly-v2-12b --num-gpus 2 --debug
CUDA_VISIBLE_DEVICES=4 python3 -m fastchat.serve.cli --model THUDM/chatglm-6b --debug
CUDA_VISIBLE_DEVICES=5 python3 -m fastchat.serve.cli --model lmsys/fastchat-t5-3b-v1.0 --debug
CUDA_VISIBLE_DEVICES=6 python3 -m fastchat.serve.cli --model ~/model_weights/baize-7b --debug
CUDA_VISIBLE_DEVICES=7 python3 -m fastchat.serve.cli --model ~/model_weights/RWKV-4-Raven-7B-v11x-Eng99%-Other1%-20230429-ctx8192.pth --debug
```

### Test GUI Serving

```
python3 -m fastchat.serve.controller
```

```
CUDA_VISIBLE_DEVICES=0,1 python3 -m fastchat.serve.model_worker --model-path ~/model_weights/koala-13b --num-gpus 2 --port 30000 --worker http://localhost:30000
CUDA_VISIBLE_DEVICES=2,3 python3 -m fastchat.serve.model_worker --model-path ~/model_weights/alpaca-13b --num-gpus 2 --port 30002 --worker http://localhost:30002
CUDA_VISIBLE_DEVICES=4,5 python3 -m fastchat.serve.model_worker --model-path ~/model_weights/vicuna-13b --port 30004 --worker http://localhost:30004 --num-gpus 2
CUDA_VISIBLE_DEVICES=6,7 python3 -m fastchat.serve.model_worker --model-path OpenAssistant/oasst-sft-1-pythia-12b --port 30006 --worker http://localhost:30006 --num-gpus 2

CUDA_VISIBLE_DEVICES=0,1 python3 -m fastchat.serve.model_worker --model-path StabilityAI/stablelm-tuned-alpha-7b --num-gpus 2 --port 30000 --worker http://localhost:30000
CUDA_VISIBLE_DEVICES=2,3 python3 -m fastchat.serve.model_worker --model-path databricks/dolly-v2-12b --num-gpus 2 --port 30002 --worker http://localhost:30002
CUDA_VISIBLE_DEVICES=4 python3 -m fastchat.serve.model_worker --model-path THUDM/chatglm-6b --port 30004 --worker http://localhost:30004
CUDA_VISIBLE_DEVICES=5 python3 -m fastchat.serve.model_worker --model-path lmsys/fastchat-t5-3b-v1.0 --port 30005 --worker http://localhost:30005
CUDA_VISIBLE_DEVICES=6 python3 -m fastchat.serve.model_worker --model-path ~/model_weights/baize-7b --port 30006 --worker http://localhost:30006
CUDA_VISIBLE_DEVICES=7 python3 -m fastchat.serve.model_worker --model-path ~/model_weights/RWKV-4-Raven-7B-v11x-Eng99%-Other1%-20230429-ctx8192.pth --port 30007 --worker http://localhost:30007
```

```
python3 -m fastchat.serve.gradio_web_server_multi
```


### Test OpenAI API Server

```
python3 -m fastchat.serve.model_worker --model-name 'vicuna-7b-v1.1' --model-path ~/model_weights/vicuna-7b-v1.1
```

```
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
```

```
cd tests
python3 test_openai_sdk.py
bash test_openai_curl.sh
```
