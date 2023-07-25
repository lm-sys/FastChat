# AWQ 4bit Inference

We integrated [AWQ](https://github.com/mit-han-lab/llm-awq) into FastChat to provide **efficient and accurate** 4bit LLM inference.

## Install AWQ

Setup environment (please refer to [this link](https://github.com/mit-han-lab/llm-awq#install) for more details):
```bash
conda create -n fastchat-awq python=3.10 -y
conda activate fastchat-awq
# cd /path/to/FastChat
pip install --upgrade pip   # enable PEP 660 support
pip install -e .			# install fastchat

git clone https://github.com/mit-han-lab/llm-awq repositories/llm-awq
cd repositories/llm-awq
pip install -e .			# install awq package

cd awq/kernels				
python setup.py install		# install awq CUDA kernels
```

## Quantize models with AWQ

Several sample scripts for AWQ quantization are provided [here](https://github.com/mit-han-lab/llm-awq/tree/main/scripts). An example for quantizing LLaMA2-7b-chat is as follows.

1. Perform AWQ search and save search results (the search results for many commonly used models are provided in [AWQ model zoo](https://github.com/mit-han-lab/llm-awq/tree/main#awq-model-zoo)):

```bash
python -m awq.entry --model_path /PATH/TO/llama2-hf/llama-2-7b-chat \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/llama-2-7b-chat-w4-g128.pt
```

2. Generate real quantized weights (INT4)

```bash
mkdir quant_cache
python -m awq.entry --model_path /PATH/TO/llama2-hf/llama-2-7b-chat \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/llama-2-7b-chat-w4-g128.pt \
    --q_backend real --dump_quant quant_cache/llama-2-7b-chat-w4-g128-awq.pt
```

* AWQ also supports evaluating the accuracy of quantization. Learn more from [this link](https://github.com/mit-han-lab/llm-awq/tree/main#usage).

## Chat with the CLI

```bash
python3 -m fastchat.serve.cli \
    --model-path /PATH/TO/llama2-hf/llama-2-7b-chat \ 	  # Path to the Hugging Face repo/model
    --awq-wbits 4 \
    --awq-groupsize 128 \
    --awq-ckpt quant_cache/llama-2-7b-chat-w4-g128-awq.pt # Path to the AWQ quantized checkpoint
```

* Note that we only use the Hugging Face directory to configure our model. And the FP16 weights will not be loaded. You can also copy the AWQ quantized checkpoint to the Hugging Face directory. In that case, there is no need to specify `--awq-ckpt`.



## Benchmark (TBD)

| LLaMA-13B | branch                 | Bits | group-size | memory(MiB) | PPL(c4) | Median(s/token) | act-order | speed up |
| --------- | ---------------------- | ---- | ---------- | ----------- | ------- | --------------- | --------- | -------- |
| FP16      | fastest-inference-4bit | 16   | -          | 26634       | 6.96    | 0.0383          | -         | 1x       |
| GPTQ      | triton                 | 4    | 128        | 8590        | 6.97    | 0.0551          | -         | 0.69x    |
| GPTQ      | fastest-inference-4bit | 4    | 128        | 8699        | 6.97    | 0.0429          | true      | 0.89x    |
| GPTQ      | fastest-inference-4bit | 4    | 128        | 8699        | 7.03    | 0.0287          | false     | 1.33x    |
| GPTQ      | fastest-inference-4bit | 4    | -1         | 8448        | 7.12    | 0.0284          | false     | 1.44x    |
