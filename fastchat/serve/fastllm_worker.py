"""
A model worker that executes the model based on fastllm.

https://github.com/ztxz16/fastllm

Code based on vllm_worker.py and mlx_worker.py

You must install fastllm:

    git clone https://github.com/ztxz16/fastllm
    cd fastllm
    mkdir build && cd build
    cmake .. -DUSE_CUDA=ON  # or OFF for CPU-only
    make -j
    cd ../
    cd pyfastllm && python setup.py install
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


def _load_fastllm_model(model_path, dtype="float16", threads=4):
    """Load model using fastllm. Supports HuggingFace and .flm format."""
    try:
        from ftllm import llm
    except ImportError:
        raise ImportError(
            "fastllm is not installed. Please install it from "
            "https://github.com/ztxz16/fastllm"
        )
    model = llm.model(model_path, dtype=dtype)
    model.set_threads(threads)
    return model


class FastLLMWorker(BaseModelWorker):
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
        model,
        context_len: int,
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
            f"worker type: fastllm worker..."
        )
        self.model = model
        self.context_len = context_len
        self.tokenizer = None

        if not no_register:
            self.init_heart_beat()

    def count_token(self, params):
        prompt = params["prompt"]
        # fastllm models expose a response_token_count or we estimate by chars
        # Use a rough estimate: ~1.3 tokens per word for Chinese/English mixed
        token_count = max(1, len(prompt) // 2)
        return {"count": token_count, "error_code": 0}

    async def generate_stream(self, params):
        self.call_ct += 1

        context = params.pop("prompt")
        request_id = params.pop("request_id")
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = int(params.get("top_k", 1))
        max_new_tokens = int(params.get("max_new_tokens", 256))
        stop_str = params.get("stop", None)
        stop_token_ids = params.get("stop_token_ids", None) or []
        echo = params.get("echo", True)
        repetition_penalty = float(params.get("repetition_penalty", 1.0))

        # Handle stop_str
        stop = set()
        if isinstance(stop_str, str) and stop_str != "":
            stop.add(stop_str)
        elif isinstance(stop_str, list) and stop_str != []:
            stop.update(stop_str)

        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            top_p = 1.0

        generated_text = ""
        completion_tokens = 0
        finish_reason = None

        # fastllm stream_response yields token strings one at a time
        def _stream_tokens():
            return self.model.stream_response(
                context,
                max_length=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repeat_penalty=repetition_penalty,
            )

        iterator = await run_in_threadpool(_stream_tokens)

        for token_text in iterator:
            generated_text += token_text
            completion_tokens += 1

            # Check for stop strings
            stopped = False
            for s in stop:
                if s in generated_text:
                    generated_text = generated_text[: generated_text.index(s)]
                    stopped = True
                    break

            if stopped:
                finish_reason = "stop"
                break

            partial_stop = any(is_partial_stop(generated_text, i) for i in stop)
            if partial_stop:
                continue

            if completion_tokens >= max_new_tokens:
                finish_reason = "length"

            output_text = context + generated_text if echo else generated_text
            ret = {
                "text": output_text,
                "error_code": 0,
                "usage": {
                    "prompt_tokens": max(1, len(context) // 2),
                    "completion_tokens": completion_tokens,
                    "total_tokens": max(1, len(context) // 2) + completion_tokens,
                },
                "cumulative_logprob": [],
                "finish_reason": finish_reason,
            }

            if finish_reason is not None:
                yield (
                    json.dumps({**ret, "finish_reason": None}) + "\0"
                ).encode()
            yield (json.dumps(ret) + "\0").encode()

            if finish_reason is not None:
                break

        # If loop ended without explicit finish_reason
        if finish_reason is None:
            finish_reason = "stop"
            output_text = context + generated_text if echo else generated_text
            ret = {
                "text": output_text,
                "error_code": 0,
                "usage": {
                    "prompt_tokens": max(1, len(context) // 2),
                    "completion_tokens": completion_tokens,
                    "total_tokens": max(1, len(context) // 2) + completion_tokens,
                },
                "cumulative_logprob": [],
                "finish_reason": None,
            }
            yield (json.dumps(ret) + "\0").encode()
            ret["finish_reason"] = finish_reason
            yield (json.dumps(ret) + "\0").encode()

    async def generate(self, params):
        async for x in self.generate_stream(params):
            pass
        return json.loads(x[:-1].decode())


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
    request_id = str(uuid.uuid4())
    params["request_id"] = request_id
    generator = worker.generate_stream(params)
    background_tasks = create_background_tasks(request_id)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = str(uuid.uuid4())
    params["request_id"] = request_id
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
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the model (HuggingFace format or .flm file)")
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--limit-worker-concurrency", type=int, default=1024)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument(
        "--conv-template", type=str, default=None,
        help="Conversation prompt template.",
    )
    parser.add_argument(
        "--dtype", type=str, default="float16",
        choices=["float16", "float32", "int8", "int4"],
        help="Data type for model weights (default: float16)",
    )
    parser.add_argument(
        "--threads", type=int, default=4,
        help="Number of CPU threads for fastllm (default: 4)",
    )
    parser.add_argument(
        "--context-length", type=int, default=2048,
        help="Maximum context length (default: 2048)",
    )

    args = parser.parse_args()

    model = _load_fastllm_model(args.model_path, dtype=args.dtype, threads=args.threads)

    worker = FastLLMWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        args.no_register,
        args.conv_template,
        model,
        args.context_length,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
