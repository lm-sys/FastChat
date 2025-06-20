"""
A model worker that executes the TensorRT engine.

Refer to the implemention in https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/run.py
"""

import argparse
import asyncio
import json
import os
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

import re
import torch
from transformers import AutoTokenizer, T5Tokenizer

import tensorrt_llm
from tensorrt_llm import runtime
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner
from tensorrt_llm.builder import get_engine_version
from copy import deepcopy

from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.serve.model_worker import (
    logger,
    worker_id,
)
from fastchat.utils import get_context_length, is_partial_stop


app = FastAPI()


def read_model_name(engine_dir: str):
    engine_version = get_engine_version(engine_dir)

    with open(Path(engine_dir) / "config.json", "r") as f:
        config = json.load(f)

    if engine_version is None:
        return config["builder_config"]["name"]

    return config["pretrained_config"]["architecture"]


def throttle_generator(generator, stream_interval):
    for i, out in enumerate(generator):
        if not i % stream_interval:
            yield out

    if i % stream_interval:
        yield out


def load_tokenizer(
    tokenizer_dir: Optional[str] = None,
    vocab_file: Optional[str] = None,
    model_name: str = "gpt",
    tokenizer_type: Optional[str] = None,
):
    if vocab_file is None:
        use_fast = True
        if tokenizer_type is not None and tokenizer_type == "llama":
            use_fast = False
        # Should set both padding_side and truncation_side to be 'left'
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            legacy=False,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=True,
            tokenizer_type=tokenizer_type,
            use_fast=use_fast,
        )
    else:
        # For gpt-next, directly load from tokenizer.model
        assert model_name == "gpt"
        tokenizer = T5Tokenizer(
            vocab_file=vocab_file, padding_side="left", truncation_side="left"
        )

    if model_name == "qwen":
        with open(Path(tokenizer_dir) / "generation_config.json") as f:
            gen_config = json.load(f)
        chat_format = gen_config["chat_format"]
        if chat_format == "raw":
            pad_id = gen_config["pad_token_id"]
            end_id = gen_config["eos_token_id"]
        elif chat_format == "chatml":
            pad_id = tokenizer.im_end_id
            end_id = tokenizer.im_end_id
        else:
            raise Exception(f"unknown chat format: {chat_format}")
    elif model_name == "glm_10b":
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eop_token_id
    else:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eos_token_id

    return tokenizer, pad_id, end_id


class TensorRTWorker(BaseModelWorker):
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
        runner: ModelRunner,
        tokenizer: AutoTokenizer,
        pad_id: int,
        end_id: int,
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
            f"Loading the model {self.model_names} on worker {worker_id}."
            f"worker type: tensorRT worker..."
        )
        logger.info(
            (
                "worker args:\n"
                f"controller_addr: {controller_addr}\n"
                f"worker_addr: {worker_addr}\n"
                f"worker_id: {worker_id}\n"
                f"model_path: {model_path}\n"
                f"model_names: {model_names}\n"
                f"limit_worker_concurrency: {limit_worker_concurrency}\n"
                f"no_register: {no_register}\n"
                f"conv_template: {conv_template}\n"
                f"runner: {runner}\n"
                f"tokenizer: {tokenizer}\n"
                f"pad_id: {pad_id}\n"
                f"end_id: {end_id}\n"
            )
        )

        self.runner = runner
        self.tokenizer = tokenizer
        self.pad_id = pad_id
        self.end_id = end_id
        self.context_len = get_context_length(self.runner.config)

        if not no_register:
            self.init_heart_beat()

    async def generate_stream(self, params):
        self.call_ct += 1
        try:

            def generate(runner, tokenizer, params):
                input_ids = tokenizer.encode(
                    params["prompt"], add_special_tokens=True, truncation=True
                )
                batch_input_ids = [torch.tensor(input_ids, dtype=torch.int32)]

                max_new_tokens = int(params.get("max_new_tokens", 128))
                temperature = float(params.get("temperature", 0.7))
                top_k = int(params.get("top_k", -1))
                top_p = float(params.get("top_p", 1.0))

                assert top_k != -1, "Top_k in TensorRT should not be -1"

                outputs = runner.generate(
                    batch_input_ids,
                    max_new_tokens=max_new_tokens,
                    end_id=self.end_id,
                    pad_id=self.pad_id,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    num_beams=1,
                    streaming=True,
                    output_sequence_lengths=True,
                    return_dict=True,
                )
                torch.cuda.synchronize()

                input_lengths = [x.size(0) for x in batch_input_ids]
                for curr_outputs in throttle_generator(outputs, 1):
                    if tensorrt_llm.mpi_rank() == 0:
                        output_ids = curr_outputs["output_ids"]
                        sequence_lengths = curr_outputs["sequence_lengths"]
                        batch_size, num_beams, _ = output_ids.size()
                        for batch_idx in range(0, batch_size):
                            for beam in range(num_beams):
                                output_begin = input_lengths[batch_idx]
                                output_end = sequence_lengths[batch_idx][beam].item()
                                outputs = output_ids[batch_idx][beam][
                                    output_begin:output_end
                                ].tolist()
                                output_text = tokenizer.decode(outputs)
                                response = output_text
                                yield {
                                    "text": response,
                                    "usage": {
                                        "prompt_tokens": input_lengths[batch_idx],
                                        "completion_tokens": output_end - output_begin,
                                        "total_tokens": input_lengths[batch_idx]
                                        + output_end
                                        - output_begin,
                                    },
                                    "finish_reason": None,
                                }
                yield {
                    "text": response,
                    "usage": {
                        "prompt_tokens": input_lengths[0],
                        "completion_tokens": output_end - output_begin,
                        "total_tokens": input_lengths[0] + output_end - output_begin,
                    },
                    "finish_reason": "stop",
                }

            for output in generate(self.runner, self.tokenizer, params):
                ret = {
                    "text": output["text"],
                    "error_code": 0,
                }
                if "usage" in output:
                    ret["usage"] = output["usage"]
                if "finish_reason" in output:
                    ret["finish_reason"] = output["finish_reason"]
                if "logprobs" in output:
                    ret["logprobs"] = output["logprobs"]
                yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
            yield json.dumps(ret).encode() + b"\0"
        except (ValueError, RuntimeError) as e:
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


def create_model_worker():
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
        help="tensorRT engine path",
        default="lmsys/vicuna-7b-v1.5",
    )
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--tokenizer-path", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--limit-worker-concurrency", type=int, default=5)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--lora-dir",
        type=str,
        default=None,
        nargs="+",
        help="The directory of LoRA weights",
    )
    parser.add_argument(
        "--debug-mode",
        default=False,
        action="store_true",
        help="Whether or not to turn on the debug mode",
    )

    args = parser.parse_args()
    logger.info(f"args: {args}")

    # load tokenizer
    model_name = read_model_name(args.model_path)
    tokenizer, pad_id, end_id = load_tokenizer(
        tokenizer_dir=args.tokenizer_path, model_name=model_name
    )

    # load trt runner
    runtime_rank = tensorrt_llm.mpi_rank()
    runner_cls = ModelRunner
    runner_kwargs = dict(
        engine_dir=args.model_path,
        lora_dir=args.lora_dir,
        rank=runtime_rank,
        debug_mode=args.debug_mode,
    )
    runner = runner_cls.from_dir(**runner_kwargs)

    # get config
    config_path = os.path.join(args.model_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    runner.config = {
        **config["build_config"],
        **config["build_config"]["plugin_config"],
    }

    # create worker
    worker = TensorRTWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        model_path=args.model_path,
        model_names=args.model_names,
        limit_worker_concurrency=args.limit_worker_concurrency,
        no_register=args.no_register,
        conv_template=args.conv_template,
        runner=runner,
        tokenizer=tokenizer,
        pad_id=pad_id,
        end_id=end_id,
    )

    return args, worker


if __name__ == "__main__":
    args, worker = create_model_worker()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
