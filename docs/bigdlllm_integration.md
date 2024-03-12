# BigDL-LLM integration
You can use [BigDL-LLM](https://github.com/intel-analytics/BigDL) as an optimized worker implementation in FastChat.
It offers the functionality to accelerate LLM inference on Intel CPU and GPU using INT4/FP4/INT8/FP8 with very low latency.
See the supported models [here](https://github.com/intel-analytics/BigDL?tab=readme-ov-file#verified-models).

## Instruction
1. Install BigDL-LLM.

You can install BigDL-LLM in Linux environment using the following guide:

```bash
# Recommended to use conda environment
# For CPU installment:
pip install --pre --upgrade bigdl-llm[all]

# For XPU installment:
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
```

> For a more detailed steps on how to install `BigDL-LLM`, you can refer to this [documentation](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install.html)


2. When you launch a model worker, replace the normal worker (`fastchat.serve.model_worker`) with the vLLM worker (`fastchat.serve.bigdl_worker`). All other commands such as controller, gradio web server, and OpenAI API server are kept the same.

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

