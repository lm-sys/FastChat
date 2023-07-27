# AWQ 4bit Inference

We integrated [AWQ](https://github.com/mit-han-lab/llm-awq) into FastChat to provide **efficient and accurate** 4bit LLM inference.

## Install AWQ

Setup environment (please refer to [this link](https://github.com/mit-han-lab/llm-awq#install) for more details):
```bash
conda create -n fastchat-awq python=3.10 -y
conda activate fastchat-awq
# cd /path/to/FastChat
pip install --upgrade pip    # enable PEP 660 support
pip install -e .             # install fastchat

git clone https://github.com/mit-han-lab/llm-awq repositories/llm-awq
cd repositories/llm-awq
pip install -e .             # install awq package

cd awq/kernels				
python setup.py install	     # install awq CUDA kernels
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

## Benchmark

* Through **4-bit weight quantization**, AWQ helps to run larger language models within the device memory restriction and prominently accelerates token generation. All benchmarks are done with group_size 128. 

* Benchmark on NVIDIA RTX A6000:

  | Model           | Bits | Max Memory (MiB) | Speed (ms/token) | AWQ Speedup |
  | --------------- | ---- | ---------------- | ---------------- | ----------- |
  | vicuna-7b       | 16   | 13543            | 26.06            | /           |
  | vicuna-7b       | 4    | 5547             | 12.43            | 2.1x        |
  | llama2-7b-chat  | 16   | 13543            | 27.14            | /           |
  | llama2-7b-chat  | 4    | 5547             | 12.44            | 2.2x        |
  | vicuna-13b      | 16   | 25647            | 44.91            | /           |
  | vicuna-13b      | 4    | 9355             | 17.30            | 2.6x        |
  | llama2-13b-chat | 16   | 25647            | 47.28            | /           |
  | llama2-13b-chat | 4    | 9355             | 20.28            | 2.3x        |

* NVIDIA RTX 4090:

  | Model           | AWQ 4bit Speed (ms/token) | FP16 Speed (ms/token) | AWQ Speedup |
  | --------------- | ------------------------- | --------------------- | ----------- |
  | vicuna-7b       | 8.61                      | 19.09                 | 2.2x        |
  | llama2-7b-chat  | 8.66                      | 19.97                 | 2.3x        |
  | vicuna-13b      | 12.17                     | OOM                   | /           |
  | llama2-13b-chat | 13.54                     | OOM                   | /           |

* NVIDIA Jetson Orin:

  | Model           | AWQ 4bit Speed (ms/token) | FP16 Speed (ms/token) | AWQ Speedup |
  | --------------- | ------------------------- | --------------------- | ----------- |
  | vicuna-7b       | 65.34                     | 93.12                 | 1.4x        |
  | llama2-7b-chat  | 75.11                     | 104.71                | 1.4x        |
  | vicuna-13b      | 115.40                    | OOM                   | /           |
  | llama2-13b-chat | 136.81                    | OOM                   | /           |
