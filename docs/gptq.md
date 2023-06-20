# GPTQ 4bit Inference

Support GPTQ 4bit inference with [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa).

1. Window user: use the `old-cuda` branch.
2. Linux user: recommend the `fastest-inference-4bit` branch.

## Install

Setup environment:
```bash
# cd /path/to/FastChat
git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa.git repositories/GPTQ-for-LLaMa
cd repositories/GPTQ-for-LLaMa
# Window's user should use the `old-cuda` branch
git switch fastest-inference-4bit
# Install `quant-cuda` package in FastChat's virtualenv
python3 setup_cuda.py install
pip3 install texttable
```

Chat with the CLI:
```bash
python3 -m fastchat.serve.cli \
    --model-path models/vicuna-7B-1.1-GPTQ-4bit-128g \
    --gptq-wbits 4 \
    --gptq-groupsize 128
```

Start model worker:
```bash
# Download quantized model from huggingface
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/TheBloke/vicuna-7B-1.1-GPTQ-4bit-128g models/vicuna-7B-1.1-GPTQ-4bit-128g

python3 -m fastchat.serve.model_worker \
    --model-path models/vicuna-7B-1.1-GPTQ-4bit-128g \
    --gptq-wbits 4 \
    --gptq-groupsize 128

# You can specify which quantized model to use
python3 -m fastchat.serve.model_worker \
    --model-path models/vicuna-7B-1.1-GPTQ-4bit-128g \
    --gptq-ckpt models/vicuna-7B-1.1-GPTQ-4bit-128g/vicuna-7B-1.1-GPTQ-4bit-128g.safetensors \
    --gptq-wbits 4 \
    --gptq-groupsize 128 \
    --gptq-act-order
```

## Benchmark

| LLaMA-13B | branch                 | Bits | group-size | memory(MiB) | PPL(c4) | Median(s/token) | act-order | speed up |
| --------- | ---------------------- | ---- | ---------- | ----------- | ------- | --------------- | --------- | -------- |
| FP16      | fastest-inference-4bit | 16   | -          | 26634       | 6.96    | 0.0383          | -         | 1x       |
| GPTQ      | triton                 | 4    | 128        | 8590        | 6.97    | 0.0551          | -         | 0.69x    |
| GPTQ      | fastest-inference-4bit | 4    | 128        | 8699        | 6.97    | 0.0429          | true      | 0.89x    |
| GPTQ      | fastest-inference-4bit | 4    | 128        | 8699        | 7.03    | 0.0287          | false     | 1.33x    |
| GPTQ      | fastest-inference-4bit | 4    | -1         | 8448        | 7.12    | 0.0284          | false     | 1.44x    |
