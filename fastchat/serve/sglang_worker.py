"""
A model worker that executes the model based on SGLANG.

Usage:
python3 -m fastchat.serve.sglang_worker --model-path liuhaotian/llava-v1.5-7b --tokenizer-path llava-hf/llava-1.5-7b-hf --port 30000 --worker-address http://localhost:30000
"""

import argparse
import asyncio
import json
import multiprocessing
from typing import List

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import sglang as sgl
from sglang.srt.hf_transformers_utils import get_tokenizer, get_config
from sglang.srt.utils import load_image

from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.serve.model_worker import (
    logger,
    worker_id,
)
from fastchat.utils import get_context_length, is_partial_stop

app = FastAPI()


@sgl.function
def pipeline(s, prompt, max_tokens):
    for p in prompt:
        if isinstance(p, str):
            s += p
        else:
            s += sgl.image(p)
    s += sgl.gen("response", max_tokens=max_tokens)


class SGLWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        tokenizer_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        conv_template: str,
        runtime: sgl.Runtime,
        trust_remote_code: bool,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template,
        )

        logger.info(
            f"Loading the model {self.model_names} on worker {worker_id}, worker type: SGLang worker..."
        )

        self.tokenizer = get_tokenizer(tokenizer_path)
        self.context_len = get_context_length(
            get_config(model_path, trust_remote_code=trust_remote_code)
        )

        if not no_register:
            self.init_heart_beat()

    async def generate_stream_gate(self, params):
        self.call_ct += 1

        prompt = params.pop("prompt")
        images = params.get("images", [])
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = params.get("top_k", -1.0)
        frequency_penalty = float(params.get("frequency_penalty", 0.0))
        presence_penalty = float(params.get("presence_penalty", 0.0))
        max_new_tokens = params.get("max_new_tokens", 256)
        stop_str = params.get("stop", None)
        stop_token_ids = params.get("stop_token_ids", None) or []
        echo = params.get("echo", True)

        # Handle stop_str
        stop = []
        if isinstance(stop_str, str) and stop_str != "":
            stop.append(stop_str)
        elif isinstance(stop_str, list) and stop_str != []:
            stop.extend(stop_str)

        for tid in stop_token_ids:
            if tid is not None:
                s = self.tokenizer.decode(tid)
                if s != "":
                    stop.append(s)

        # make sampling params for sgl.gen
        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            top_p = 1.0

        # split prompt by image token
        split_prompt = prompt.split("<image>\n")
        if prompt.count("<image>\n") != len(images):
            raise ValueError(
                "The number of images passed in does not match the number of <image> tokens in the prompt!"
            )
        prompt = []
        for i in range(len(split_prompt)):
            prompt.append(split_prompt[i])
            if i < len(images):
                prompt.append(load_image(images[i]))

        state = pipeline.run(
            prompt,
            max_new_tokens,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stream=True,
        )

        entire_output = prompt if echo else ""
        async for out, meta_info in state.text_async_iter(
            var_name="response", return_meta_data=True
        ):
            partial_stop = any(is_partial_stop(out, i) for i in stop)

            # prevent yielding partial stop sequence
            if partial_stop:
                continue

            entire_output += out
            prompt_tokens = meta_info["prompt_tokens"]
            completion_tokens = meta_info["completion_tokens"]

            ret = {
                "text": entire_output,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "error_code": 0,
            }

            yield (json.dumps(ret) + "\0").encode()

    async def generate_gate(self, params):
        async for x in self.generate_stream_gate(params):
            pass
        return json.loads(x[:-1].decode())


def release_worker_semaphore():
    worker.semaphore.release()


def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()


def create_background_tasks():
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    return background_tasks


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    generator = worker.generate_stream_gate(params)
    background_tasks = create_background_tasks()
    return StreamingResponse(generator)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    output = await worker.generate_gate(params)
    release_worker_semaphore()
    return JSONResponse(output)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    return worker.count_token(params)


@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    return worker.get_conv_template()


@app.post("/model_details")
async def api_model_details(request: Request):
    return {"context_length": worker.context_len}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--model-path", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--tokenizer-path", type=str, default="")
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--limit-worker-concurrency", type=int, default=1024)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_false",
        default=True,
        help="Trust remote code (e.g., from HuggingFace) when"
        "downloading the model and tokenizer.",
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.9,
        help="The ratio (between 0 and 1) of GPU memory to"
        "reserve for the model weights, activations, and KV cache. Higher"
        "values will increase the KV cache size and thus improve the model's"
        "throughput. However, if the value is too high, it may cause out-of-"
        "memory (OOM) errors.",
    )

    args = parser.parse_args()

    args.tp_size = args.num_gpus if args.num_gpus > 1 else 1
    args.tokenizer_path = (
        args.model_path if args.tokenizer_path == "" else args.tokenizer_path
    )

    multiprocessing.set_start_method("spawn", force=True)
    runtime = sgl.Runtime(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        trust_remote_code=args.trust_remote_code,
        mem_fraction_static=args.mem_fraction_static,
        tp_size=args.tp_size,
        log_level="info",
    )
    sgl.set_default_backend(runtime)

    worker = SGLWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.tokenizer_path,
        args.model_names,
        args.limit_worker_concurrency,
        args.no_register,
        args.conv_template,
        runtime,
        args.trust_remote_code,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
