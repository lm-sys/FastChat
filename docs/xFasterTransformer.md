# xFasterTransformer Inference Franework

Integrated [xFasterTransformer](https://github.com/intel/xFasterTransformer) customized framework into Fastchat to provide **Faster** inference speed on Intel CPU.

## Install xFasterTransformer

Setup environment (please refer to [this link](https://github.com/intel/xFasterTransformer#installation) for more details):

```bash
pip install xfastertransformer
```

Prepare Model (please refer to [this link](https://github.com/intel/xFasterTransformer#prepare-model) for more details):
```bash
python ./tools/chatglm_convert.py -i ${HF_DATASET_DIR} -o  ${OUTPUT_DIR}
```

Chat with the CLI:
```bash
#run inference on all CPUs and using float16
python3 -m fastchat.serve.cli \
    --model-path /path/to/models \
    --enable-xft --xft-dtype fp16
```
or with numactl on multi-socket server for better performance
```bash
#run inference on numanode 0 and with data type bf16_fp16 (first token uses bfloat16, and rest tokens use float16)
numactl -N 0  --localalloc python3 -m fastchat.serve.cli --model-path /path/to/models/chatglm2_6b_cpu/ --enable-xft --xft-dtype bf16_fp16
```

Start model worker:
```bash
# Load model with default configuration (max sequence length 4096, no GPU split setting).
python3 -m fastchat.serve.model_worker \
    --model-path /path/to/models \
    --enable-xft \
    --xft-dtype bf16_fp16 \

```

For more details, please refer to [this link](https://github.com/intel/xFasterTransformer#how-to-run) 
