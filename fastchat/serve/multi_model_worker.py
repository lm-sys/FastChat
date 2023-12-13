"""
A multi-model worker that contains multiple sub-works one for each model.  This
supports running a list of models on the same machine so that they can
(potentially) share the same background weights.

Each model can have one or more model names.

This multi-model worker assumes the models shares some underlying weights and
thus reports the combined queue lengths for health checks.

We recommend using this with multiple Peft models (with `peft` in the name)
where all Peft models are trained on the exact same base model.
"""
import argparse
import asyncio
import dataclasses
import logging
import json
import os
import sys
import time
from typing import List, Union
import threading
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import requests

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
import uvicorn

from collections import deque

from fastchat.constants import WORKER_HEART_BEAT_INTERVAL, ErrorCode, SERVER_ERROR_MSG
from fastchat.model.model_adapter import (
    load_model,
    add_model_args,
    get_conversation_template,
)
from fastchat.model.model_chatglm import generate_stream_chatglm
from fastchat.model.model_falcon import generate_stream_falcon
from fastchat.model.model_codet5p import generate_stream_codet5p
from fastchat.modules.gptq import GptqConfig
from fastchat.modules.exllama import ExllamaConfig
from fastchat.modules.xfastertransformer import XftConfig
from fastchat.serve.inference import generate_stream
from fastchat.serve.model_worker import ModelWorker, worker_id, logger
from fastchat.utils import build_logger, pretty_print_semaphore, get_context_length


lazy_loading_enabled = False

# We store both the underlying workers and a mapping from their model names to
# the worker instance.  This makes it easy to fetch the appropriate worker for
# each API call.
workers = []
worker_map = {}
# The active_workers queue is used with lazy loading
active_workers = deque([])
app = FastAPI()


def release_worker_semaphore(model_name: str = ""):
    if lazy_loading_enabled and model_name:
        worker = worker_map[model_name]
        if worker.is_model_loaded():
            worker.unload_model()
            worker.semaphore.release()
    else:  # Not using lazy-loading
        workers[0].semaphore.release()


async def acquire_worker_semaphore(model_name: str = ""):
    if workers[0].semaphore is None:
        # Share the same semaphore for all workers because
        # all workers share the same GPU.
        semaphore = asyncio.BoundedSemaphore(workers[0].limit_worker_concurrency)
        for w in workers:
            w.semaphore = semaphore
    if lazy_loading_enabled and model_name:
        worker = worker_map[model_name]

        # Make sure this worker is at the front of the queue because it's the
        # most recently accessed. If this worker was not originally in the
        # queue, this may push a worker off the back of the queue.
        if worker in active_workers:
            active_workers.remove(worker)
        active_workers.appendleft(worker)

        if not worker.is_model_loaded():  # We don't have the semaphore
            for mn, w in worker_map.items():
                if w not in active_workers:
                    release_worker_semaphore(mn)
            await worker.semaphore.acquire()
            worker.lazy_load_model()
    else:  # Not using lazy-loading
        await workers[0].semaphore.acquire()


def create_background_tasks():
    background_tasks = BackgroundTasks()
    if not lazy_loading_enabled:
        background_tasks.add_task(release_worker_semaphore)
    return background_tasks


# Note: for all the calls below, we make a hard assumption that the caller
# includes the model name in the payload, otherwise we can't figure out which
# underlying sub-worker to call.


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore(model_name=params["model"])
    worker = worker_map[params["model"]]
    generator = None
    if lazy_loading_enabled:
        # Streaming doesn't work with lazy-loading because the generator keeps
        # a reference to the model and prevents it from being unloaded. In
        # order to preserve compatibility with the streaming interface, we copy
        # the whole output from the stream and use an iterator on the copy
        # instead.
        stream_copy = list(worker.generate_stream_gate(params))
        generator = iter(stream_copy)
    else:
        generator = worker.generate_stream_gate(params)
    background_tasks = create_background_tasks()
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore(model_name=params["model"])
    worker = worker_map[params["model"]]
    output = worker.generate_gate(params)
    if not lazy_loading_enabled:
        release_worker_semaphore()
    return JSONResponse(output)


@app.post("/worker_get_embeddings")
async def api_get_embeddings(request: Request):
    params = await request.json()
    await acquire_worker_semaphore(model_name=params["model"])
    worker = worker_map[params["model"]]
    embedding = worker.get_embeddings(params)
    background_tasks = create_background_tasks()
    return JSONResponse(content=embedding, background=background_tasks)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return {
        "model_names": [m for w in workers for m in w.model_names],
        "speed": 1,
        "queue_length": sum([w.get_queue_length() for w in workers]),
    }


@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    if lazy_loading_enabled:
        await acquire_worker_semaphore(model_name=params["model"])
    worker = worker_map[params["model"]]
    return worker.count_token(params)


@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    params = await request.json()
    if lazy_loading_enabled:
        await acquire_worker_semaphore(model_name=params["model"])
    worker = worker_map[params["model"]]
    return worker.get_conv_template()


@app.post("/model_details")
async def api_model_details(request: Request):
    params = await request.json()
    if lazy_loading_enabled:
        await acquire_worker_semaphore(model_name=params["model"])
    worker = worker_map[params["model"]]
    return {"context_length": worker.context_len}


def create_multi_model_worker():
    # Note: Ensure we resolve arg conflicts.  We let `add_model_args` add MOST
    # of the model args but we'll override one to have an append action that
    # supports multiple values.
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    add_model_args(parser)
    # Override the model path to be repeated and align it with model names.
    parser.add_argument(
        "--model-path",
        type=str,
        default=[],
        action="append",
        help="One or more paths to model weights to load. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        action="append",
        help="One or more model names.  Values must be aligned with `--model-path` values.",
    )
    parser.add_argument(
        "--conv-template",
        type=str,
        default=None,
        action="append",
        help="Conversation prompt template. Values must be aligned with `--model-path` values. If only one value is provided, it will be repeated for all models.",
    )
    parser.add_argument(
        "--lazy",
        action="store_true",
        required=False,
        default=False,
        help="Enable lazy-loading of models.",
    )
    parser.add_argument(
        "--limit-worker-concurrency", type=int, default=1 if "--lazy" in sys.argv else 5
    )
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument(
        "--ssl",
        action="store_true",
        required=False,
        default=False,
        help="Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'.",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    gptq_config = GptqConfig(
        ckpt=args.gptq_ckpt or args.model_path,
        wbits=args.gptq_wbits,
        groupsize=args.gptq_groupsize,
        act_order=args.gptq_act_order,
    )
    if args.enable_exllama:
        exllama_config = ExllamaConfig(
            max_seq_len=args.exllama_max_seq_len,
            gpu_split=args.exllama_gpu_split,
            cache_8bit=args.exllama_cache_8bit,
        )
    else:
        exllama_config = None
    if args.enable_xft:
        xft_config = XftConfig(
            max_seq_len=args.xft_max_seq_len,
            data_type=args.xft_dtype,
        )
        if args.device != "cpu":
            print("xFasterTransformer now is only support CPUs. Reset device to CPU")
            args.device = "cpu"
    else:
        xft_config = None

    if args.model_names is None:
        args.model_names = [[x.split("/")[-1]] for x in args.model_path]

    if args.conv_template is None:
        args.conv_template = [None] * len(args.model_path)
    elif len(args.conv_template) == 1:  # Repeat the same template
        args.conv_template = args.conv_template * len(args.model_path)

    # Launch all workers
    workers = []
    for conv_template, model_path, model_names in zip(
        args.conv_template, args.model_path, args.model_names
    ):
        w = ModelWorker(
            args.controller_address,
            args.worker_address,
            worker_id,
            model_path,
            model_names,
            args.limit_worker_concurrency,
            args.no_register,
            device=args.device,
            num_gpus=args.num_gpus,
            max_gpu_memory=args.max_gpu_memory,
            load_8bit=args.load_8bit,
            cpu_offloading=args.cpu_offloading,
            gptq_config=gptq_config,
            exllama_config=exllama_config,
            xft_config=xft_config,
            stream_interval=args.stream_interval,
            conv_template=conv_template,
            lazy=args.lazy,
        )
        workers.append(w)
        for model_name in model_names:
            worker_map[model_name] = w

    # Register all models
    url = args.controller_address + "/register_worker"
    data = {
        "worker_name": workers[0].worker_addr,
        "check_heart_beat": not args.no_register,
        "worker_status": {
            "model_names": [m for w in workers for m in w.model_names],
            "speed": 1,
            "queue_length": sum([w.get_queue_length() for w in workers]),
        },
    }
    r = requests.post(url, json=data)
    assert r.status_code == 200

    return args, workers


if __name__ == "__main__":
    args, workers = create_multi_model_worker()

    lazy_loading_enabled = args.lazy
    if lazy_loading_enabled:
        # args.limit_worker_concurrency controls how many models at a time will
        # be loaded into VRAM.
        active_workers = deque([], args.limit_worker_concurrency)
        # Preload up to n=limit_worker_concurrency models into VRAM.
        for wkr in workers[: args.limit_worker_concurrency]:
            asyncio.run(acquire_worker_semaphore(wkr.model_names[0]))

    if args.ssl:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            ssl_keyfile=os.environ["SSL_KEYFILE"],
            ssl_certfile=os.environ["SSL_CERTFILE"],
        )
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
