"""
A model worker that calls huggingface inference endpoint.

Register models in a JSON file with the following format:
{
    "falcon-180b-chat": {
        "model_name": "falcon-180B-chat",
        "api_base": "https://api-inference.huggingface.co/models",
        "model_path": "tiiuae/falcon-180B-chat",
        "token": "hf_XXX",
        "context_length": 2048
    },
    "zephyr-7b-beta": {
        "model_name": "zephyr-7b-beta",
        "model_path": "",
        "api_base": "xxx",
        "token": "hf_XXX",
        "context_length": 4096
    }
}

"model_path", "api_base", "token", and "context_length" are necessary, while others are optional.
"""
import argparse
import asyncio
import json
import uuid
import os
from typing import List, Optional

import requests
import uvicorn
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from huggingface_hub import InferenceClient

from fastchat.constants import SERVER_ERROR_MSG, ErrorCode
from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.utils import build_logger

worker_id = str(uuid.uuid4())[:8]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")

workers = []
worker_map = {}
app = FastAPI()


# reference to
# https://github.com/philschmid/easyllm/blob/cbd908b3b3f44a97a22cb0fc2c93df3660bacdad/easyllm/clients/huggingface.py#L374-L392
def get_gen_kwargs(
    params,
    seed: Optional[int] = None,
):
    stop = params.get("stop", None)
    if isinstance(stop, list):
        stop_sequences = stop
    elif isinstance(stop, str):
        stop_sequences = [stop]
    else:
        stop_sequences = []
    gen_kwargs = {
        "do_sample": True,
        "return_full_text": bool(params.get("echo", False)),
        "max_new_tokens": int(params.get("max_new_tokens", 256)),
        "top_p": float(params.get("top_p", 1.0)),
        "temperature": float(params.get("temperature", 1.0)),
        "stop_sequences": stop_sequences,
        "repetition_penalty": float(params.get("repetition_penalty", 1.0)),
        "top_k": params.get("top_k", None),
        "seed": seed,
    }
    if gen_kwargs["top_p"] == 1:
        gen_kwargs["top_p"] = 0.9999999
    if gen_kwargs["top_p"] == 0:
        gen_kwargs.pop("top_p")
    if gen_kwargs["temperature"] == 0:
        gen_kwargs.pop("temperature")
        gen_kwargs["do_sample"] = False
    return gen_kwargs


def could_be_stop(text, stop):
    for s in stop:
        if any(text.endswith(s[:i]) for i in range(1, len(s) + 1)):
            return True
    return False


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
        conv_template: Optional[str] = None,
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
        self.seed = seed

        logger.info(
            f"Connecting with huggingface api {self.model_path} as {self.model_names} on worker {worker_id} ..."
        )

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

        prompt = params["prompt"]
        gen_kwargs = get_gen_kwargs(params, seed=self.seed)
        stop = gen_kwargs["stop_sequences"]
        if "falcon" in self.model_path and "chat" in self.model_path:
            stop.extend(["\nUser:", "<|endoftext|>", " User:", "###"])
            stop = list(set(stop))
            gen_kwargs["stop_sequences"] = stop

        logger.info(f"prompt: {prompt}")
        logger.info(f"gen_kwargs: {gen_kwargs}")

        try:
            if self.model_path == "":
                url = f"{self.api_base}"
            else:
                url = f"{self.api_base}/{self.model_path}"
            client = InferenceClient(url, token=self.token)
            res = client.text_generation(
                prompt, stream=True, details=True, **gen_kwargs
            )

            reason = None
            text = ""
            for chunk in res:
                if chunk.token.special:
                    continue
                text += chunk.token.text

                s = next((x for x in stop if text.endswith(x)), None)
                if s is not None:
                    text = text[: -len(s)]
                    reason = "stop"
                    break
                if could_be_stop(text, stop):
                    continue
                if (
                    chunk.details is not None
                    and chunk.details.finish_reason is not None
                ):
                    reason = chunk.details.finish_reason
                if reason not in ["stop", "length"]:
                    reason = None
                ret = {
                    "text": text,
                    "error_code": 0,
                    "finish_reason": reason,
                }
                yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
            yield json.dumps(ret).encode() + b"\0"

    def generate_gate(self, params):
        for x in self.generate_stream_gate(params):
            pass
        return json.loads(x[:-1].decode())

    def get_embeddings(self, params):
        raise NotImplementedError()


def release_worker_semaphore(worker):
    worker.semaphore.release()


def acquire_worker_semaphore(worker):
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()


def create_background_tasks(worker):
    background_tasks = BackgroundTasks()
    background_tasks.add_task(lambda: release_worker_semaphore(worker))
    return background_tasks


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    worker = worker_map[params["model"]]
    await acquire_worker_semaphore(worker)
    generator = worker.generate_stream_gate(params)
    background_tasks = create_background_tasks(worker)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    worker = worker_map[params["model"]]
    await acquire_worker_semaphore(worker)
    output = worker.generate_gate(params)
    release_worker_semaphore(worker)
    return JSONResponse(output)


@app.post("/worker_get_embeddings")
async def api_get_embeddings(request: Request):
    params = await request.json()
    worker = worker_map[params["model"]]
    await acquire_worker_semaphore(worker)
    embedding = worker.get_embeddings(params)
    release_worker_semaphore(worker)
    return JSONResponse(content=embedding)


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
    worker = worker_map[params["model"]]
    return worker.count_token(params)


@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    params = await request.json()
    worker = worker_map[params["model"]]
    return worker.get_conv_template()


@app.post("/model_details")
async def api_model_details(request: Request):
    params = await request.json()
    worker = worker_map[params["model"]]
    return {"context_length": worker.context_len}


def create_huggingface_api_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    # all model-related parameters are listed in --model-info-file
    parser.add_argument(
        "--model-info-file",
        type=str,
        required=True,
        help="Huggingface API model's info file path",
    )

    parser.add_argument(
        "--limit-worker-concurrency",
        type=int,
        default=5,
        help="Limit the model concurrency to prevent OOM.",
    )
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Overwrite the random seed for each generation.",
    )
    parser.add_argument(
        "--ssl",
        action="store_true",
        required=False,
        default=False,
        help="Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'.",
    )
    args = parser.parse_args()

    with open(args.model_info_file, "r", encoding="UTF-8") as f:
        model_info = json.load(f)

    logger.info(f"args: {args}")

    model_path_list = []
    api_base_list = []
    token_list = []
    context_length_list = []
    model_names_list = []
    conv_template_list = []

    for m in model_info:
        model_path_list.append(model_info[m]["model_path"])
        api_base_list.append(model_info[m]["api_base"])
        token_list.append(model_info[m]["token"])

        context_length = model_info[m]["context_length"]
        model_names = model_info[m].get("model_names", [m.split("/")[-1]])
        if isinstance(model_names, str):
            model_names = [model_names]
        conv_template = model_info[m].get("conv_template", None)

        context_length_list.append(context_length)
        model_names_list.append(model_names)
        conv_template_list.append(conv_template)

    logger.info(f"Model paths: {model_path_list}")
    logger.info(f"API bases: {api_base_list}")
    logger.info(f"Tokens: {token_list}")
    logger.info(f"Context lengths: {context_length_list}")
    logger.info(f"Model names: {model_names_list}")
    logger.info(f"Conv templates: {conv_template_list}")

    for (
        model_names,
        conv_template,
        model_path,
        api_base,
        token,
        context_length,
    ) in zip(
        model_names_list,
        conv_template_list,
        model_path_list,
        api_base_list,
        token_list,
        context_length_list,
    ):
        m = HuggingfaceApiWorker(
            args.controller_address,
            args.worker_address,
            worker_id,
            model_path,
            api_base,
            token,
            context_length,
            model_names,
            args.limit_worker_concurrency,
            no_register=args.no_register,
            conv_template=conv_template,
            seed=args.seed,
        )
        workers.append(m)
        for name in model_names:
            worker_map[name] = m

    # register all the models
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
    args, workers = create_huggingface_api_worker()
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
