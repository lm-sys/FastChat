"""
A model worker that executes the model based on dash-infer.

See documentations at docs/dashinfer_integration.md
"""

import argparse
import asyncio
import copy
import json
import os
import subprocess
from typing import List

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from dashinfer.helper import EngineHelper, ConfigManager

from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.serve.model_worker import (
    logger,
    worker_id,
)
from fastchat.utils import build_logger, get_context_length, is_partial_stop


app = FastAPI()


def download_model(model_id, revision):
    source = "huggingface"
    if os.environ.get("FASTCHAT_USE_MODELSCOPE", "False").lower() == "true":
        source = "modelscope"

    logger.info(f"Downloading model {model_id} (revision: {revision}) from {source}")
    if source == "modelscope":
        from modelscope import snapshot_download

        model_dir = snapshot_download(model_id, revision=revision)
    elif source == "huggingface":
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(repo_id=model_id)
    else:
        raise ValueError("Unknown source")

    logger.info(f"Save model to path {model_dir}")

    return model_dir


class DashInferWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        revision: str,
        no_register: bool,
        config: json,
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
            f"Loading the model {self.model_names} on worker {worker_id}, worker type: dash-infer worker..."
        )
        # check if model_path is existed at local path
        if not os.path.exists(model_path):
            model_path = download_model(model_path, revision)
        engine_helper = EngineHelper(config)
        engine_helper.init_tokenizer(model_path)
        engine_helper.convert_model(model_path)
        engine_helper.init_engine()

        self.context_len = engine_helper.engine_config["engine_max_length"]
        self.tokenizer = engine_helper.tokenizer
        self.engine_helper = engine_helper

        if not no_register:
            self.init_heart_beat()

    async def generate_stream(self, params):
        self.call_ct += 1

        context = params.pop("prompt")
        temperature = params.get("temperature")
        top_k = params.get("top_k")
        top_p = params.get("top_p")
        repetition_penalty = params.get("repetition_penalty")
        presence_penalty = params.get("presence_penalty")
        max_new_tokens = params.get("max_new_tokens")
        stop_token_ids = params.get("stop_token_ids") or []
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.append(self.tokenizer.eos_token_id)
        seed = params.get("seed")
        echo = params.get("echo", True)
        logprobs = params.get("logprobs")
        # not supported parameters
        frequency_penalty = params.get("frequency_penalty")
        stop = params.get("stop")
        use_beam_search = params.get("use_beam_search", False)
        best_of = params.get("best_of", None)

        gen_cfg = copy.deepcopy(self.engine_helper.default_gen_cfg) or dict()
        if temperature is not None:
            gen_cfg["temperature"] = float(temperature)
        if top_k is not None:
            dashinfer_style_top_k = 0 if int(top_k) == -1 else int(top_k)
            gen_cfg["top_k"] = dashinfer_style_top_k
        if top_p is not None:
            gen_cfg["top_p"] = float(top_p)
        if repetition_penalty is not None:
            gen_cfg["repetition_penalty"] = float(repetition_penalty)
        if presence_penalty is not None:
            gen_cfg["presence_penalty"] = float(presence_penalty)
        if len(stop_token_ids) != 0:
            dashinfer_style_stop_token_ids = [[id] for id in set(stop_token_ids)]
            logger.info(
                f"dashinfer_style_stop_token_ids = {dashinfer_style_stop_token_ids}"
            )
            gen_cfg["stop_words_ids"] = dashinfer_style_stop_token_ids
        if seed is not None:
            gen_cfg["seed"] = int(seed)
        if logprobs is not None:
            gen_cfg["logprobs"] = True
            gen_cfg["top_logprobs"] = int(logprobs)
        if frequency_penalty is not None:
            logger.warning(
                "dashinfer worker does not support `frequency_penalty` parameter"
            )
        if stop is not None:
            logger.warning("dashinfer worker does not support `stop` parameter")
        if use_beam_search == True:
            logger.warning(
                "dashinfer worker does not support `use_beam_search` parameter"
            )
        if best_of is not None:
            logger.warning("dashinfer worker does not support `best_of` parameter")

        logger.info(
            f"dashinfer engine helper creates request with context: {context}, gen_cfg: {gen_cfg}"
        )

        request_list = self.engine_helper.create_request([context], gen_cfg=[gen_cfg])

        engine_req = request_list[0]

        # check if prompt tokens exceed the max_tokens
        max_tokens = (
            gen_cfg["max_length"]
            if max_new_tokens is None
            else engine_req.in_tokens_len + max_new_tokens
        )
        if engine_req.in_tokens_len > max_tokens:
            ret = {
                "text": f"This model's maximum generated tokens include context are {max_tokens}, However, your context resulted in {engine_req.in_tokens_len} tokens",
                "error_code": ErrorCode.CONTEXT_OVERFLOW,
            }
            yield json.dumps(ret).encode() + b"\0"
        else:
            gen_cfg["max_length"] = int(max_tokens)
            logger.info(
                f"dashinfer is going to process one request in stream mode: {engine_req}"
            )
            results_generator = self.engine_helper.process_one_request_stream(
                engine_req
            )

            try:
                for generate_text in results_generator:
                    if echo:
                        output_text = context + generate_text
                    else:
                        output_text = generate_text
                    prompt_tokens = engine_req.in_tokens_len
                    completion_tokens = engine_req.out_tokens_len
                    ret = {
                        "text": output_text,
                        "error_code": 0,
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    }
                    yield (json.dumps(ret) + "\0").encode()
            except Exception as e:
                ret = {
                    "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                    "error_code": ErrorCode.INTERNAL_ERROR,
                }
                yield json.dumps(ret).encode() + b"\0"

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


def create_background_tasks():
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    return background_tasks


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    generator = worker.generate_stream(params)
    background_tasks = create_background_tasks()
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--model-path", type=str, default="qwen/Qwen-7B-Chat")
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )

    parser.add_argument("--limit-worker-concurrency", type=int, default=1024)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Hugging Face Hub model revision identifier",
    )
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "config_file",
        metavar="config-file",
        type=str,
        default="config_qwen_v10_7b.json",
        help="A model config file which dash-inferread",
    )

    args = parser.parse_args()
    config = ConfigManager.get_config_from_json(args.config_file)

    cmd = f"pip show dashinfer | grep 'Location' | cut -d ' ' -f 2"
    package_location = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True
    )
    package_location = package_location.stdout.strip()
    os.environ["AS_DAEMON_PATH"] = package_location + "/dashinfer/allspark/bin"
    os.environ["AS_NUMA_NUM"] = str(len(config["device_ids"]))
    os.environ["AS_NUMA_OFFSET"] = str(config["device_ids"][0])
    worker = DashInferWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        args.revision,
        args.no_register,
        config,
        args.conv_template,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
