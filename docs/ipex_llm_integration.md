# IPEX-LLM integration
You can use [IPEX-LLM](https://github.com/intel-analytics/ipex-llm) as an optimized worker implementation in FastChat.
It offers the functionality to accelerate LLM inference on Intel CPU and GPU using INT4/FP4/INT8/FP8/FP16 etc. with very low latency.
See the supported models [here](https://github.com/intel-analytics/BigDL?tab=readme-ov-file#verified-models).

## Instruction
1. Install IPEX-LLM.

You can install BigDL-LLM in Linux environment using the following guide:

```bash
# Recommended to use conda environment
# For CPU installment:
pip install --pre --upgrade ipex-llm[all]

# For XPU installment:
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

> For a more detailed steps on how to install `IPEX-LLM`, you can refer to this [documentation](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_linux_gpu.html#install-ipex-llm-on-linux-with-intel-gpu)


2. When you launch a model worker, replace the normal worker (`fastchat.serve.model_worker`) with the `ipex_llm` worker (`fastchat.serve.ipex_llm_worker`). All other commands such as controller, gradio web server, and OpenAI API server are kept the same.

For CPU example:

```python
# Recommended settings
source bigdl-llm-init -t

# Available low_bit format including sym_int4, sym_int8, bf16 etc.
python3 -m fastchat.serve.bigdl_worker --model-path lmsys/vicuna-7b-v1.5 --low-bit "sym_int4" --trust-remote-code --device "cpu"
```

For GPU example:

```python
python3 -m fastchat.serve.bigdl_worker --model-path lmsys/vicuna-7b-v1.5 --low-bit "sym_int4" --trust-remote-code --device "xpu"
```

We have also provided an option `--load-low-bit-model` to load models that have been converted and saved into disk using the `save_low_bit` interface as introduced in this [document](https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/example/CPU/HF-Transformers-AutoModels/Save-Load/README.md).

Check the following examples:
```bash
# Or --device "cpu"
python -m ipex_llm.serving.fastchat.ipex_llm_worker --model-path /Low/Bit/Model/Path --trust-remote-code --device "xpu" --load-low-bit-model
```


## Using self-speculative decoding:

You can use IPEX-LLM to run `self-speculative decoding` example. Refer to [here](https://github.com/intel-analytics/ipex-llm/tree/c9fac8c26bf1e1e8f7376fa9a62b32951dd9e85d/python/llm/example/GPU/Speculative-Decoding) for more details on intel MAX GPUs. Refer to [here](https://github.com/intel-analytics/ipex-llm/tree/c9fac8c26bf1e1e8f7376fa9a62b32951dd9e85d/python/llm/example/GPU/Speculative-Decoding) for more details on intel CPUs.

```bash
# Available low_bit format only including bf16 on CPU.
source ipex-llm-init -t
python -m ipex_llm.serving.fastchat.ipex_llm_worker --model-path lmsys/vicuna-7b-v1.5 --low-bit "bf16" --trust-remote-code --device "cpu" --speculative

# Available low_bit format only including fp16 on GPU.
source /opt/intel/oneapi/setvars.sh
export ENABLE_SDP_FUSION=1
export SYCL_CACHE_PERSISTENT=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
python -m ipex_llm.serving.fastchat.ipex_llm_worker --model-path lmsys/vicuna-7b-v1.5 --low-bit "fp16" --trust-remote-code --device "xpu" --speculative
```

For a full list of accepted arguments, you can refer to the main method of the `ipex_llm_worker.py`


