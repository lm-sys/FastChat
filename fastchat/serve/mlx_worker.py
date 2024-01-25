"""
A model worker using Apple MLX

https://github.com/ml-explore/mlx-examples/tree/main/llms

Code based on vllm_worker https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/vllm_worker.py

You must install MLX python:

pip install mlx-lm
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
from fastchat.utils import get_context_length, is_partial_stop

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.utils import generate_step

app = FastAPI()


class MLXWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        llm_engine: "MLX",
        conv_template: str,
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
            f"Loading the model {self.model_names} on worker {worker_id}, worker type: MLX worker..."
        )

        self.model_name = model_path
        self.mlx_model, self.mlx_tokenizer = load(model_path)

        self.tokenizer = self.mlx_tokenizer
        # self.context_len = get_context_length(
        #     llm_engine.engine.model_config.hf_config)
        self.context_len = 2048  # hard code for now -- not sure how to get in MLX

        if not no_register:
            self.init_heart_beat()

    async def generate_stream(self, params):
        self.call_ct += 1

        context = params.pop("prompt")
        request_id = params.pop("request_id")
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = params.get("top_k", -1.0)
        presence_penalty = float(params.get("presence_penalty", 0.0))
        frequency_penalty = float(params.get("frequency_penalty", 0.0))
        max_new_tokens = params.get("max_new_tokens", 256)
        stop_str = params.get("stop", None)
        stop_token_ids = params.get("stop_token_ids", None) or []
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.append(self.tokenizer.eos_token_id)
        echo = params.get("echo", True)
        use_beam_search = params.get("use_beam_search", False)
        best_of = params.get("best_of", None)

        # Handle stop_str
        stop = set()
        if isinstance(stop_str, str) and stop_str != "":
            stop.add(stop_str)
        elif isinstance(stop_str, list) and stop_str != []:
            stop.update(stop_str)

        for tid in stop_token_ids:
            if tid is not None:
                s = self.tokenizer.decode(tid)
                if s != "":
                    stop.add(s)

        print("Stop patterns: ", stop)

        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            top_p = 1.0

        tokens = []
        skip = 0

        context_mlx = mx.array(self.tokenizer.encode(context))

        finish_reason = "length"

        iterator = await run_in_threadpool(
            generate_step, context_mlx, self.mlx_model, temperature
        )

        for i in range(max_new_tokens):
            token = await run_in_threadpool(next, iterator)
            if token == self.mlx_tokenizer.eos_token_id:
                finish_reason = "stop"
                break
            tokens.append(token.item())
            tokens_decoded = self.mlx_tokenizer.decode(tokens)
            last_token_decoded = self.mlx_tokenizer.decode([token.item()])
            skip = len(tokens_decoded)

            partial_stop = any(is_partial_stop(tokens_decoded, i) for i in stop)

            if partial_stop:
                finish_reason = "stop"
                break

            ret = {
                "text": tokens_decoded,
                "error_code": 0,
                "usage": {
                    "prompt_tokens": len(context),
                    "completion_tokens": len(tokens),
                    "total_tokens": len(context) + len(tokens),
                },
                "cumulative_logprob": [],
                "finish_reason": None,  # hard code for now
            }
            # print(ret)
            yield (json.dumps(ret) + "\0").encode()
        ret = {
            "text": self.mlx_tokenizer.decode(tokens),
            "error_code": 0,
            "usage": {},
            "cumulative_logprob": [],
            "finish_reason": finish_reason,
        }
        yield (json.dumps(obj={**ret, **{"finish_reason": None}}) + "\0").encode()
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
        print("trying to abort but not implemented")

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
    # await engine.abort(request_id)
    print("Trying to abort but not implemented")
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
    parser.add_argument("--model-path", type=str, default="microsoft/phi-2")
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_false",
        default=True,
        help="Trust remote code (e.g., from HuggingFace) when"
        "downloading the model and tokenizer.",
    )

    args, unknown = parser.parse_known_args()

    if args.model_path:
        args.model = args.model_path

    worker = MLXWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        1024,
        False,
        "MLX",
        args.conv_template,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
