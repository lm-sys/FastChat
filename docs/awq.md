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

## Chat with the CLI

```bash
# Download quantized model from huggingface
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/mit-han-lab/vicuna-7b-v1.3-4bit-g128-awq

# You can specify which quantized model to use by setting --awq-ckpt
python3 -m fastchat.serve.cli \
    --model-path models/vicuna-7b-v1.3-4bit-g128-awq \
    --awq-wbits 4 \
    --awq-groupsize 128 
```

## Benchmark (TBD)
