### Local GPU cluster
node-01
```
python3 -m fastchat.serve.controller --host 0.0.0.0 --port 10002

CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.vllm_worker --model-path lmsys/vicuna-13b-v1.5 --model-name vicuna-13b --controller http://node-01:10002 --host 0.0.0.0 --port 31000 --worker-address http://$(hostname):31000
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.vllm_worker --model-path lmsys/vicuna-13b-v1.5 --model-name vicuna-13b --controller http://node-01:10002 --host 0.0.0.0 --port 31001 --worker-address http://$(hostname):31001

CUDA_VISIBLE_DEVICES=2,3 ray start --head
python3 -m fastchat.serve.vllm_worker --model-path lmsys/vicuna-33b-v1.3 --model-name vicuna-33b --controller http://node-01:10002 --host 0.0.0.0 --port 31002 --worker-address http://$(hostname):31002 --num-gpus 2
```

node-02
```
CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.vllm_worker --model-path meta-llama/Llama-2-13b-chat-hf --model-name llama-2-13b-chat --controller http://node-01:10002 --host 0.0.0.0 --port 31000 --worker-address http://$(hostname):31000 --tokenizer meta-llama/Llama-2-7b-chat-hf
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.vllm_worker --model-path meta-llama/Llama-2-13b-chat-hf --model-name llama-2-13b-chat --controller http://node-01:10002 --host 0.0.0.0 --port 31001 --worker-address http://$(hostname):31001 --tokenizer meta-llama/Llama-2-7b-chat-hf
CUDA_VISIBLE_DEVICES=2 python3 -m fastchat.serve.vllm_worker --model-path meta-llama/Llama-2-7b-chat-hf --model-name llama-2-7b-chat --controller http://node-01:10002 --host 0.0.0.0 --port 31002 --worker-address http://$(hostname):31002 --tokenizer meta-llama/Llama-2-7b-chat-hf
CUDA_VISIBLE_DEVICES=3 python3 -m fastchat.serve.vllm_worker --model-path WizardLM/WizardLM-13B-V1.1 --model-name wizardlm-13b  --controller http://node-01:10002 --host 0.0.0.0 --port 31003 --worker-address http://$(hostname):31003
```

node-03
```
python3 -m fastchat.serve.vllm_worker --model-path mosaicml/mpt-30b-chat --controller http://node-01:10002 --host 0.0.0.0 --port 31000 --worker-address http://$(hostname):31000 --num-gpus 2
python3 -m fastchat.serve.vllm_worker --model-path timdettmers/guanaco-33b-merged --model-name guanaco-33b  --controller http://node-01:10002 --host 0.0.0.0 --port 31002 --worker-address http://$(hostname):31002 --num-gpus 2 --tokenizer hf-internal-testing/llama-tokenizer
```

node-04
```
CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.multi_model_worker --model-path ~/model_weights/RWKV-4-Raven-14B-v12-Eng98%25-Other2%25-20230523-ctx8192.pth --model-name RWKV-4-Raven-14B --model-path lmsys/fastchat-t5-3b-v1.0 --model-name fastchat-t5-3b --controller http://node-01:10002 --host 0.0.0.0 --port 31000 --worker http://$(hostname):31000 --limit 4
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.multi_model_worker --model-path OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5 --model-name oasst-pythia-12b --model-path mosaicml/mpt-7b-chat --model-name mpt-7b-chat --controller http://node-01:10002 --host 0.0.0.0 --port 31001 --worker http://$(hostname):31001 --limit 4
CUDA_VISIBLE_DEVICES=2 python3 -m fastchat.serve.multi_model_worker --model-path lmsys/vicuna-7b-v1.5 --model-name vicuna-7b --model-path THUDM/chatglm-6b --model-name chatglm-6b --controller http://node-01:10002 --host 0.0.0.0 --port 31002 --worker http://$(hostname):31002 --limit 4
CUDA_VISIBLE_DEVICES=3 python3 -m fastchat.serve.vllm_worker --model-path ~/model_weights/alpaca-13b  --controller http://node-01:10002 --host 0.0.0.0 --port 31003 --worker-address http://$(hostname):31003
```

test
```
python3 -m fastchat.serve.test_message --model vicuna-13b --controller http://localhost:10002
```
