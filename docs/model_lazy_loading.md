# Model lazy-loading with `multi_model_worker` #

The `multi_model_worker` now supports lazy-loading models. This enables you to run a `multi_model_worker` with more models than can fit in your VRAM.

Use the `--lazy` command-line parameter to enable lazy-loading mode. Using the `--limit-worker-concurrency` parameter, you can specify how many of your chosen models to load into VRAM at any one time. When using lazy-loading mode, `--limit-worker-concurrency` defaults to 1.

When using lazy-loading mode, a model is loaded into VRAM when it is accessed.

## Example: 3 unquantized models, loaded into VRAM one at a time ##

Let's say you want to be able run three 13B models, but you only have the VRAM to load one at a time.

```bash
python3 -m fastchat.serve.multi_model_worker \
    --lazy --limit-worker-concurrency 1 \
	--model-path lmsys/vicuna-13b-v1.5 \
	--model-path lmsys/longchat-13b-16k \
	--model-path NousResearch/Nous-Hermes-Llama2-13b
```

This will pre-load `vicuna-13b-v1.5` into VRAM. The following sequence of events illustrates the lazy-loading functionality:

1. A query is sent to `vicuna-13b-v1.5`: the model is already in VRAM, so no models are unloaded or loaded
2. a query is sent to `longchat-13b-16k`: the model is not currently in VRAM, so `vicuna-13b-v1.5` is unloaded and then `longchat-13b-16k` is loaded, then the query response is returned
3. a query is sent to `Nous-Hermes-Llama2-13b`: the model is not currently in VRAM, so `longchat-13b-16k` is unloaded and then `Nous-Hermes-Llama2-13b` is loaded, then the query response is returned
4. a query is sent to `vicuna-13b-v1.5`: the model is not currently in VRAM, so `Nous-Hermes-Llama2-13b` is unloaded and then `vicuna-13b-v1.5` is loaded, then the query response is returned.

## Example: 3 quantized models, loaded into VRAM two at a time ##

This example will use `exllama`. Loading multiple models into VRAM is also supported when using the `Transformers` backend, but this allows us to kill two birds with one section of this document. This example also mixes quantization types (`exl2` and `GPTQ`) and model sizes (13b and 7b). We'll also use `--exllama-cache-8bit` to save a bit more VRAM.

### Download the models ###

When we're not using `Transformers` we have to download the models beforehand.

Make sure you have git-lfs installed (https://git-lfs.com).
```bash
git lfs install
```

In this example, we run the following in the `models` directory.

Download `turboderp/CodeLlama-13B-instruct-exl2` with `4.65bpw` quantization:
```bash
git clone https://huggingface.co/turboderp/CodeLlama-13B-instruct-exl2 -b 4.65bpw
```

Download `latimar/Synthia-13B-exl2` with `4.625bpw` quantization:
```bash
git clone https://huggingface.co/latimar/Synthia-13B-exl2 -b 4_625-bpw-h6
```

Download `TheBloke/zephyr-7B-beta-GPTQ` with 4-bit quantization:
```bash
git clone https://huggingface.co/TheBloke/zephyr-7B-beta-GPTQ
```

### Run the `multi_model_worker` ###

```bash
python3 -m fastchat.serve.multi_model_worker \
    --lazy --limit-worker-concurrency 2 \
    --model-path ./models/CodeLlama-13B-instruct-exl2 --enable-exllama --exllama-cache-8bit \
    --model-path ./models/Synthia-13B-exl2 --enable-exllama --exllama-cache-8bit \
    --model-path ./models/zephyr-7B-beta-GPTQ --enable-exllama --exllama-cache-8bit
```

This will pre-load both `CodeLlama-13B-instruct-exl2` and `Synthia-13B-exl2` into VRAM. Note that lazy-loading uses a "last-accessed-last-out" queue. This means that, when a new model is loaded, it keeps the most-recently-accessed model(s) in VRAM (subject to the `--limit-worker-concurrency` value). The following sequence of events illustrates the lazy-loading functionality:

1. A query is sent to `CodeLlama-13B-instruct-exl2`: the model is already in VRAM, so no models are unloaded or loaded
2. A query is sent to `Synthia-13B-exl2`: the model is already in VRAM, so no models are unloaded or loaded
3. a query is sent to `zephyr-7B-beta-GPTQ`: the model is not currently in VRAM, so `CodeLlama-13B-instruct-exl2` is unloaded and then `zephyr-7B-beta-GPTQ` is loaded, then the query response is returned
4. A query is sent to `Synthia-13B-exl2`: the model is already in VRAM, so no models are unloaded or loaded
3. a query is sent to `CodeLlama-13B-instruct-exl2`: the model is not currently in VRAM, so `zephyr-7B-beta-GPTQ` is unloaded and then `CodeLlama-13B-instruct-exl2` is loaded, then the query response is returned.

## Notes ##

### If you have the option, use the non-streaming interface ###

In order to fully unload an unused model from VRAM, lazy-loading mode cannot use the real streaming mode. Lazy-loading mode supports the streaming interface by creating a copy of the model's output, and then streaming on that copy. This degrades performance, so it is recommended that you use the non-streaming interface when you can.

### Per-GPU lazy-loading `multi_model_worker`s ###

If you have multiple GPUs, you can run a lazy-loading `multi_model_worker` on each GPU. To do this, specify the GPU and ports for each worker. For instance, to run both the examples given above at the same time, you could do the following:

```bash
# worker 0
CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.multi_model_worker \
    --lazy --limit-worker-concurrency 1 \
	--model-path lmsys/vicuna-13b-v1.5 \
	--model-path lmsys/longchat-13b-16k \
	--model-path NousResearch/Nous-Hermes-Llama2-13b \
    --port 31000 --worker http://localhost:31000
# worker 1
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.multi_model_worker \
    --lazy --limit-worker-concurrency 2 \
    --model-path ./models/CodeLlama-13B-instruct-exl2 --enable-exllama --exllama-cache-8bit \
    --model-path ./models/Synthia-13B-exl2 --enable-exllama --exllama-cache-8bit \
    --model-path ./models/zephyr-7B-beta-GPTQ --enable-exllama --exllama-cache-8bit \
    --port 31001 --worker http://localhost:31001
```

This would run the unquantized models on your first GPU (worker 0), lazy-loading one at a time into VRAM. It would run the quantized models on your second GPU (worker 1), lazy-loading two at a time into VRAM.

### Aggregating GPU memory from multiple GPUs is untested with lazy-loading mode ###

The `multi_model_worker` supports GPU memory aggregation by using the `--num-gpus` parameter. However, this is untested when using lazy-loading mode, and so the behaviour is undefined.
