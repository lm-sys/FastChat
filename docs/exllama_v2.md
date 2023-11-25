# ExllamaV2 GPTQ Inference Framework

Integrated [ExllamaV2](https://github.com/turboderp/exllamav2) customized kernel into Fastchat to provide **Faster** GPTQ inference speed.

**Note: Exllama not yet support embedding REST API.**

## Install ExllamaV2

Setup environment (please refer to [this link](https://github.com/turboderp/exllamav2#how-to) for more details):

```bash
git clone https://github.com/turboderp/exllamav2
cd exllamav2
pip install -e .
```

Chat with the CLI:
```bash
python3 -m fastchat.serve.cli \
    --model-path models/vicuna-7B-1.1-GPTQ-4bit-128g \
    --enable-exllama
```

Start model worker:
```bash
# Download quantized model from huggingface
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/TheBloke/vicuna-7B-1.1-GPTQ-4bit-128g models/vicuna-7B-1.1-GPTQ-4bit-128g

# Load model with default configuration (max sequence length 4096, no GPU split setting).
python3 -m fastchat.serve.model_worker \
    --model-path models/vicuna-7B-1.1-GPTQ-4bit-128g \
    --enable-exllama

#Load model with max sequence length 2048, allocate 18 GB to CUDA:0 and 24 GB to CUDA:1.
python3 -m fastchat.serve.model_worker \
    --model-path models/vicuna-7B-1.1-GPTQ-4bit-128g \
    --enable-exllama \
    --exllama-max-seq-len 2048 \
    --exllama-gpu-split 18,24
```

`--exllama-cache-8bit` can be used to enable 8-bit caching with exllama and save some VRAM.

## Performance 

Reference: https://github.com/turboderp/exllamav2#performance


| Model      | Mode         | Size  | grpsz | act | V1: 3090Ti | V1: 4090 | V2: 3090Ti | V2: 4090    |
|------------|--------------|-------|-------|-----|------------|----------|------------|-------------|
| Llama      | GPTQ         | 7B    | 128   | no  | 143 t/s    | 173 t/s  | 175 t/s    | **195** t/s |
| Llama      | GPTQ         | 13B   | 128   | no  | 84 t/s     | 102 t/s  | 105 t/s    | **110** t/s |
| Llama      | GPTQ         | 33B   | 128   | yes | 37 t/s     | 45 t/s   | 45 t/s     | **48** t/s  |
| OpenLlama  | GPTQ         | 3B    | 128   | yes | 194 t/s    | 226 t/s  | 295 t/s    | **321** t/s |
| CodeLlama  | EXL2 4.0 bpw | 34B   | -     | -   | -          | -        | 42 t/s     | **48** t/s  |
| Llama2     | EXL2 3.0 bpw | 7B    | -     | -   | -          | -        | 195 t/s    | **224** t/s |
| Llama2     | EXL2 4.0 bpw | 7B    | -     | -   | -          | -        | 164 t/s    | **197** t/s |
| Llama2     | EXL2 5.0 bpw | 7B    | -     | -   | -          | -        | 144 t/s    | **160** t/s |
| Llama2     | EXL2 2.5 bpw | 70B   | -     | -   | -          | -        | 30 t/s     | **35** t/s  |
| TinyLlama  | EXL2 3.0 bpw | 1.1B  | -     | -   | -          | -        | 536 t/s    | **635** t/s |
| TinyLlama  | EXL2 4.0 bpw | 1.1B  | -     | -   | -          | -        | 509 t/s    | **590** t/s |
