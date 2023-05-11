# Fastest GPTQ 4bit Support

Support GPTQ 4bit with [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa).

1. Window user: use the `old-cuda` branch.
2. Linux user: recommend the `fastest-inference-4bit` branch.

## Install

Follow the [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa) doc to build compressed model.

Setup environment:
```bash
# cd /path/to/FastChat
git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa.git repositories/GPTQ-for-LLaMa
cd repositories/GPTQ-for-LLaMa
# Window's user should use the `old-cuda` branch
git switch fastest-inference-4bit

# install `quant-cuda` package in FastChat's virtualenv
python setup_cuda.py install
```

Start model worker:
```bash
python -m fastchat.serve.model_worker \
    --model-path ${your_model_path} \
    --load-gptq ${your_gptq_checkpoint} \
    --wbits 4 \
    --groupsize 128
```

## Benchmark

| LLaMA-13B | branch                 | Bits | group-size | memory(MiB) | PPL(c4) | Median(s/token) | act-order | speed up |
| --------- | ---------------------- | ---- | ---------- | ----------- | ------- | --------------- | --------- | -------- |
| FP16      | fastest-inference-4bit | 16   | -          | 26634       | 6.96    | 0.0383          | -         | 1x       |
| GPTQ      | triton                 | 4    | 128        | 8590        | 6.97    | 0.0551          | -         | 0.69x    |
| GPTQ      | fastest-inference-4bit | 4    | 128        | 8699        | 6.97    | 0.0429          | true      | 0.89x    |
| GPTQ      | fastest-inference-4bit | 4    | 128        | 8699        | 7.03    | 0.0287          | false     | 1.33x    |
| GPTQ      | fastest-inference-4bit | 4    | -1         | 8448        | 7.12    | 0.0284          | false     | 1.44x    |
