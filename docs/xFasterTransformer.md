# xFasterTransformer Inference Framework

Integrated [xFasterTransformer](https://github.com/intel/xFasterTransformer) customized framework into Fastchat to provide **Faster** inference speed on Intel CPU.

## Install xFasterTransformer

Setup environment (please refer to [this link](https://github.com/intel/xFasterTransformer#installation) for more details):

```bash
pip install xfastertransformer
```

## Prepare models

Prepare Model (please refer to [this link](https://github.com/intel/xFasterTransformer#prepare-model) for more details):
```bash
python ./tools/chatglm_convert.py -i ${HF_DATASET_DIR} -o  ${OUTPUT_DIR}
```

## Parameters of xFasterTransformer
--enable-xft to enable xfastertransformer in Fastchat
--xft-max-seq-len to set the max token length the model can process. max token length include input token length.
--xft-dtype to set datatype used in xFasterTransformer for computation. xFasterTransformer can support fp32, fp16, int8, bf16 and hybrid data types like : bf16_fp16, bf16_int8. For datatype details please refer to [this link](https://github.com/intel/xFasterTransformer/wiki/Data-Type-Support-Platform)
    

Chat with the CLI:
```bash
#run inference on all CPUs and using float16
python3 -m fastchat.serve.cli \
    --model-path /path/to/models \
    --enable-xft \
    --xft-dtype fp16
```
or with numactl on multi-socket server for better performance
```bash
#run inference on numanode 0 and with data type bf16_fp16 (first token uses bfloat16, and rest tokens use float16)
numactl -N 0  --localalloc \
python3 -m fastchat.serve.cli \
    --model-path /path/to/models/chatglm2_6b_cpu/ \
    --enable-xft \
    --xft-dtype bf16_fp16
```
or using MPI to run inference on 2 sockets for better performance
```bash
#run inference on numanode 0 and 1 and with data type bf16_fp16 (first token uses bfloat16, and rest tokens use float16)
OMP_NUM_THREADS=$CORE_NUM_PER_SOCKET LD_PRELOAD=libiomp5.so mpirun \
-n 1 numactl -N 0  --localalloc \
python -m fastchat.serve.cli \ 
    --model-path /path/to/models/chatglm2_6b_cpu/ \
    --enable-xft \
    --xft-dtype bf16_fp16 : \
-n 1 numactl -N 1  --localalloc \
python -m fastchat.serve.cli \
    --model-path /path/to/models/chatglm2_6b_cpu/ \
    --enable-xft \
    --xft-dtype bf16_fp16
```


Start model worker:
```bash
# Load model with default configuration (max sequence length 4096, no GPU split setting).
python3 -m fastchat.serve.model_worker \
    --model-path /path/to/models \
    --enable-xft \
    --xft-dtype bf16_fp16 
```
or with numactl on multi-socket server for better performance
```bash
#run inference on numanode 0 and with data type bf16_fp16 (first token uses bfloat16, and rest tokens use float16)
numactl -N 0  --localalloc python3 -m fastchat.serve.model_worker \
    --model-path /path/to/models \
    --enable-xft \
    --xft-dtype bf16_fp16 
```
or using MPI to run inference on 2 sockets for better performance
```bash
#run inference on numanode 0 and 1 and with data type bf16_fp16 (first token uses bfloat16, and rest tokens use float16)
OMP_NUM_THREADS=$CORE_NUM_PER_SOCKET LD_PRELOAD=libiomp5.so mpirun \
-n 1 numactl -N 0  --localalloc  python -m fastchat.serve.model_worker \
    --model-path /path/to/models \
    --enable-xft \
    --xft-dtype bf16_fp16 : \
-n 1 numactl -N 1  --localalloc  python -m fastchat.serve.model_worker \
    --model-path /path/to/models \
    --enable-xft \
    --xft-dtype bf16_fp16 
```

For more details, please refer to [this link](https://github.com/intel/xFasterTransformer#how-to-run) 
