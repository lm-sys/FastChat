"""
A model worker to call huggingface api.
The contents in supported_models.json : 
{
    "falcon-180b": {
        "model_path": "tiiuae/falcon-180B-chat",
        "api_base": "https://api-inference.huggingface.co/models",
        "token": "hf_xxx",
        "context_length": 2048
    }
}
"""
import argparse
import asyncio
import base64
import dataclasses
import gc
import logging
import json
import os
import threading
import time
from typing import List, Optional
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from huggingface_hub import InferenceClient
import requests

from fastchat.serve.model_worker import BaseModelWorker

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LlamaTokenizer,
        AutoModel,
    )
except ImportError:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LLaMATokenizer,
        AutoModel,
    )
import torch
import torch.nn.functional as F
from transformers import set_seed
import uvicorn

from fastchat.constants import WORKER_HEART_BEAT_INTERVAL, ErrorCode, SERVER_ERROR_MSG
from fastchat.conversation import get_conv_template
from fastchat.model.model_adapter import (
    load_model,
    add_model_args,
    get_conversation_template,
    get_generate_stream_function,
)
from fastchat.modules.gptq import GptqConfig
from fastchat.modules.awq import AWQConfig
from fastchat.utils import (
    build_logger,
    pretty_print_semaphore,
    get_context_length,
    str_to_torch_dtype,
)


worker_id = str(uuid.uuid4())[:8]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")

app = FastAPI()


def get_gen_kwargs():
    # TODO
    pass


class HuggingfaceApiWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        api_base: str,
        token: str,
        context_length: int,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        stream_interval: int = 2,
        conv_template: Optional[str] = None,
        embed_in_truncate: bool = False,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template=conv_template,
        )

        self.model_path = model_path
        self.api_base = api_base
        self.token = token
        self.context_len = context_length

        logger.info(
            f"Connecting with huggingface api {self.model_path} as {self.model_names} on worker {worker_id} ..."
        )

        self.model = None
        self.tokenizer = None
        self.device = None
        self.generate_stream_func = None
        self.stream_interval = stream_interval
        self.embed_in_truncate = embed_in_truncate
        self.seed = seed

        if not no_register:
            self.init_heart_beat()

    def count_token(self, params):
        # No tokenizer here
        ret = {
            "count": 0,
            "error_code": 0,
        }
        return ret

    def generate_stream_gate(self, params):
        self.call_ct += 1

        url = f"{self.api_base}/{self.model_path}"
        client = InferenceClient(url, token=self.token)
        prompt = params["prompt"]

        raise NotImplementedError()

    def generate_gate(self, params):
        for x in self.generate_stream_gate(params):
            pass
        return json.loads(x[:-1].decode())

    @torch.inference_mode()
    def get_embeddings(self, params):
        self.call_ct += 1

        raise NotImplementedError()


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
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    output = worker.generate_gate(params)
    release_worker_semaphore()
    return JSONResponse(output)


@app.post("/worker_get_embeddings")
async def api_get_embeddings(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    embedding = worker.get_embeddings(params)
    release_worker_semaphore()
    return JSONResponse(content=embedding)


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


def create_model_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument(
        "--supported-models-file", type=str, default="supported_models.json"
    )
    parser.add_argument(
        "--model", type=str, default="falcon-180b", help="The model name to be called."
    )
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        default="falcon-180b",
        help="Optional display comma separated names",
    )
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument("--embed-in-truncate", action="store_true")
    parser.add_argument(
        "--limit-worker-concurrency",
        type=int,
        default=5,
        help="Limit the model concurrency to prevent OOM.",
    )
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Overwrite the random seed for each generation.",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    with open(args.supported_models_file, "r") as f:
        supported_models = json.load(f)

    if args.model not in supported_models:
        raise ValueError(
            f"Model {args.model} not supported. Please add it to {args.supported_models_file}."
        )

    model_path = supported_models[args.model]["model_path"]
    api_base = supported_models[args.model]["api_base"]
    token = supported_models[args.model]["token"]
    context_length = supported_models[args.model]["context_length"]

    worker = HuggingfaceApiWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        model_path,
        api_base,
        token,
        context_length,
        args.model_names,
        args.limit_worker_concurrency,
        no_register=args.no_register,
        stream_interval=args.stream_interval,
        conv_template=args.conv_template,
        embed_in_truncate=args.embed_in_truncate,
        seed=args.seed,
    )
    return args, worker


if __name__ == "__main__":
    args, worker = create_model_worker()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
