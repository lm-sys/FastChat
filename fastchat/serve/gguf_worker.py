"""
A model worker using llama-cpp-python for GGUF/GGML model inference.

https://github.com/abetlen/llama-cpp-python

Code based on mlx_worker and vllm_worker.

You must install llama-cpp-python:

    pip install llama-cpp-python

Usage:
    python3 -m fastchat.serve.gguf_worker \
        --model-path /path/to/model.gguf \
        --model-names my-gguf-model
"""

import argparse
import asyncio
import atexit
import json
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
        n_gpu_layers: int = 0,
        n_ctx: int = 2048,
        n_batch: int = 512,
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

        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for GGUF model support. "
                "Install it with: pip install llama-cpp-python"
            )

        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=n_batch,
            verbose=True,
        )

        self.context_len = n_ctx
        self.tokenizer = self

        if not no_register:
            self.init_heart_beat()

    def __call__(self, text):
        """Make this object act as a tokenizer for count_token compatibility."""
        return _TokenizeResult(self.llm.tokenize(text.encode("utf-8")))

    def num_tokens(self, text):
        """Return the number of tokens in the given text."""
        return len(self.llm.tokenize(text.encode("utf-8")))

    def decode(self, token_id):
        """Decode a single token id to string."""
        return self.llm.detokenize([token_id]).decode("utf-8", errors="replace")

    async def generate_stream(self, params):
        self.call_ct += 1

        context = params.pop("prompt")
        params.pop("request_id", None)
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = int(params.get("top_k", 40))
        max_new_tokens = int(params.get("max_new_tokens", 256))
        stop_str = params.get("stop", None)
        stop_token_ids = params.get("stop_token_ids", None) or []
        echo = params.get("echo", True)
        presence_penalty = float(params.get("presence_penalty", 0.0))
        frequency_penalty = float(params.get("frequency_penalty", 0.0))

        # Build stop list
        stop = []
        if isinstance(stop_str, str) and stop_str != "":
            stop.append(stop_str)
        elif isinstance(stop_str, list) and stop_str != []:
            stop.extend(stop_str)

        # Clamp temperature
        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            temperature = 1e-5
            top_p = 1.0

        input_tokens = self.llm.tokenize(context.encode("utf-8"))
        prompt_tokens = len(input_tokens)
        generated_text = ""
        completion_tokens = 0
        finish_reason = "length"

        stream_fn = self.llm.create_completion

        def _run_stream():
            return stream_fn(
                context,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop if stop else None,
                echo=False,
                stream=True,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )

        generator = await run_in_threadpool(_run_stream)

        for chunk in generator:
            choice = chunk["choices"][0]
            delta_text = choice.get("text", "")
            generated_text += delta_text
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


class _TokenizeResult:
    """Minimal wrapper so count_token() can access .input_ids."""

    def __init__(self, token_ids):
        self.input_ids = token_ids


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
        help="Path to the GGUF model file",
    )
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--limit-worker-concurrency", type=int, default=1024)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument(
        "--conv-template",
        type=str,
        default=None,
        help="Conversation prompt template.",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=0,
        help="Number of layers to offload to GPU. Set to -1 for all layers.",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=2048,
        help="Context window size for the model.",
    )
    parser.add_argument(
        "--n-batch",
        type=int,
        default=512,
        help="Batch size for prompt processing.",
    )

    args = parser.parse_args()

    worker = GGUFWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        args.no_register,
        args.conv_template,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
