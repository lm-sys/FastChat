"""
A model worker that executes GGUF/GGML models using llama-cpp-python.

https://github.com/abetlen/llama-cpp-python

Usage:
    python3 -m fastchat.serve.gguf_worker \
        --model-path /path/to/model.gguf \
        --model-names my-gguf-model

Requires:
    pip install llama-cpp-python
"""

import argparse
import asyncio
import atexit
import json
import os
from typing import List
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.serve.model_worker import (
    logger,
    worker_id,
)
from fastchat.utils import is_partial_stop

try:
    from llama_cpp import Llama
except ImportError:
    raise ImportError(
        "llama-cpp-python is required for GGUF model support. "
        "Install it with: pip install llama-cpp-python"
    )

app = FastAPI()


class GGUFWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        conv_template: str,
        context_len: int,
        n_gpu_layers: int,
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
            f"Loading the model {self.model_names} on worker {worker_id}, "
            f"worker type: GGUF worker..."
        )

        self.model_path = model_path
        self.context_len = context_len

        self.llm = Llama(
            model_path=model_path,
            n_ctx=context_len,
            n_gpu_layers=n_gpu_layers,
            verbose=True,
        )

        # Provide a basic tokenizer interface for count_token in BaseModelWorker
        self.tokenizer = self

        if not no_register:
            self.init_heart_beat()

    def __call__(self, text):
        """Tokenizer interface for BaseModelWorker.count_token compatibility."""
        token_ids = self.llm.tokenize(text.encode("utf-8"))
        return _TokenResult(token_ids)

    def encode(self, text):
        return self.llm.tokenize(text.encode("utf-8"))

    def decode(self, tokens):
        return self.llm.detokenize(tokens).decode("utf-8", errors="replace")

    async def generate_stream(self, params):
        self.call_ct += 1

        context = params.pop("prompt")
        request_id = params.pop("request_id")
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = int(params.get("top_k", 40))
        max_new_tokens = int(params.get("max_new_tokens", 256))
        stop_str = params.get("stop", None)
        echo = params.get("echo", True)
        presence_penalty = float(params.get("presence_penalty", 0.0))
        frequency_penalty = float(params.get("frequency_penalty", 0.0))
        repeat_penalty = float(params.get("repeat_penalty", 1.1))

        # Handle stop strings
        stop = []
        if isinstance(stop_str, str) and stop_str != "":
            stop.append(stop_str)
        elif isinstance(stop_str, list) and stop_str:
            stop.extend(stop_str)

        # Clamp parameters
        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            temperature = 1e-5
            top_p = 1.0

        input_ids = self.llm.tokenize(context.encode("utf-8"))
        prompt_tokens = len(input_ids)

        generated_text = ""
        completion_tokens = 0
        finish_reason = "length"

        stream = await run_in_threadpool(
            self.llm.create_completion,
            prompt=context,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            stop=stop if stop else None,
            echo=False,
            stream=True,
        )

        for chunk in stream:
            choice = chunk["choices"][0]
            delta = choice.get("text", "")
            generated_text += delta
            completion_tokens += 1

            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]

            partial_stop = any(
                is_partial_stop(generated_text, s) for s in stop
            )
            if partial_stop:
                continue

            output_text = (context + generated_text) if echo else generated_text

            ret = {
                "text": output_text,
                "error_code": 0,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "cumulative_logprob": [],
                "finish_reason": None,
            }
            yield (json.dumps(ret) + "\0").encode()

        output_text = (context + generated_text) if echo else generated_text
        ret = {
            "text": output_text,
            "error_code": 0,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "cumulative_logprob": [],
            "finish_reason": finish_reason,
        }
        yield (json.dumps({**ret, "finish_reason": None}) + "\0").encode()
        yield (json.dumps(ret) + "\0").encode()

    async def generate(self, params):
        async for x in self.generate_stream(params):
            pass
        return json.loads(x[:-1].decode())


class _TokenResult:
    """Minimal wrapper so that BaseModelWorker.count_token can call len(result.input_ids)."""

    def __init__(self, input_ids):
        self.input_ids = input_ids


def release_worker_semaphore():
    worker.semaphore.release()


def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()


def create_background_tasks(request_id):
    async def abort_request() -> None:
        pass

    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    background_tasks.add_task(abort_request)
    return background_tasks


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = uuid.uuid4()
    params["request_id"] = str(request_id)
    generator = worker.generate_stream(params)
    background_tasks = create_background_tasks(request_id)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = uuid.uuid4()
    params["request_id"] = str(request_id)
    output = await worker.generate(params)
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


worker = None


def cleanup_at_exit():
    global worker
    print("Cleaning up...")
    del worker


atexit.register(cleanup_at_exit)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to a GGUF model file (e.g. /path/to/model.gguf)",
    )
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--limit-worker-concurrency", type=int, default=5)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument(
        "--conv-template",
        type=str,
        default=None,
        help="Conversation prompt template.",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=2048,
        help="Context length of the model. Default: 2048.",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=0,
        help="Number of layers to offload to GPU. Default: 0 (CPU only). "
        "Set to -1 to offload all layers.",
    )

    args, unknown = parser.parse_known_args()

    worker = GGUFWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        args.no_register,
        args.conv_template,
        args.context_len,
        args.n_gpu_layers,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
