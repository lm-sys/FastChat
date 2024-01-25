"""
A model worker that executes the model based on LightLLM.

See documentations at docs/lightllm_integration.md
"""

import argparse
import asyncio
import json
import os
import torch
import uvicorn

from transformers import AutoConfig

from typing import List

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse

from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.serve.model_worker import (
    logger,
    worker_id,
)

from lightllm.server.sampling_params import SamplingParams
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.server.httpserver.manager import HttpServerManager
from lightllm.server.detokenization.manager import start_detokenization_process
from lightllm.server.router.manager import start_router_process
from lightllm.server.req_id_generator import ReqIDGenerator

from lightllm.utils.net_utils import alloc_can_use_network_port
from lightllm.utils.start_utils import start_submodule_processes
from fastchat.utils import get_context_length, is_partial_stop

app = FastAPI()
g_id_gen = ReqIDGenerator()


class LightLLMWorker(BaseModelWorker):
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
        tokenizer,
        context_len,
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
            f"Loading the model {self.model_names} on worker {worker_id}, worker type: LightLLM worker..."
        )
        self.tokenizer = tokenizer
        self.context_len = context_len

        self.is_first = True

        if not no_register:
            self.init_heart_beat()

    async def generate_stream(self, params):
        self.call_ct += 1

        prompt = params.pop("prompt")
        request_id = params.pop("request_id")
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = params.get("top_k", -1.0)
        presence_penalty = float(params.get("presence_penalty", 0.0))
        frequency_penalty = float(params.get("frequency_penalty", 0.0))
        repetition_penalty = float(params.get("repetition_penalty", 1.0))
        max_new_tokens = params.get("max_new_tokens", 256)
        echo = params.get("echo", True)
        stop_str = params.get("stop", None)
        stop_token_ids = params.get("stop_token_ids", None) or []
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.append(self.tokenizer.eos_token_id)

        request = params.get("request", None)

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

        if self.is_first:
            loop = asyncio.get_event_loop()
            loop.create_task(httpserver_manager.handle_loop())
            self.is_first = False

        # make sampling params in vllm
        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            top_p = 1.0

        sampling_params = SamplingParams(
            do_sample=temperature > 0.0,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            stop_sequences=list(stop),
        )
        sampling_params.verify()

        results_generator = httpserver_manager.generate(
            prompt, sampling_params, request_id, MultimodalParams()
        )

        completion_tokens = 0
        text_outputs = ""
        cumulative_logprob = 0.0

        async for request_output, metadata, finish_status in results_generator:
            text_outputs += request_output
            completion_tokens += 1

            partial_stop = any(is_partial_stop(text_outputs, i) for i in stop)
            # prevent yielding partial stop sequence
            if partial_stop:
                continue

            if type(finish_status) is bool:  # compatibility with old version
                finish_reason = "stop" if finish_status else None
            else:
                finish_reason = finish_status.get_finish_reason()

            if request and await request.is_disconnected():
                await httpserver_manager.abort(request_id)
                finish_reason = "abort"

            logprob = metadata.get("logprob", None)
            if logprob is not None:
                cumulative_logprob += logprob

            prompt_tokens = metadata["prompt_tokens"]
            ret = {
                "text": prompt + text_outputs if echo else text_outputs,
                "error_code": 0,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "cumulative_logprob": cumulative_logprob,
            }

            if finish_reason is not None:
                yield (
                    json.dumps({**ret, "finish_reason": None}, ensure_ascii=False)
                    + "\0"
                ).encode("utf-8")
            yield (
                json.dumps({**ret, "finish_reason": finish_reason}, ensure_ascii=False)
                + "\0"
            ).encode("utf-8")

            if finish_reason is not None:  # In case of abort, we need to break the loop
                break

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
        await httpserver_manager.abort(request_id)

    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    background_tasks.add_task(abort_request)
    return background_tasks


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = g_id_gen.generate_id()
    params["request_id"] = request_id
    params["request"] = request
    generator = worker.generate_stream(params)
    background_tasks = create_background_tasks(request_id)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = g_id_gen.generate_id()
    params["request_id"] = request_id
    params["request"] = request
    output = await worker.generate(params)
    release_worker_semaphore()
    await httpserver_manager.abort(request_id)
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
    torch.multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)

    parser.add_argument(
        "--model-path",
        dest="model_dir",
        type=str,
        default=None,
        help="the model weight dir path, the app will load config, weights and tokenizer from this dir",
    )
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--limit-worker-concurrency", type=int, default=1024)
    parser.add_argument("--no-register", action="store_true")

    parser.add_argument(
        "--tokenizer_mode",
        type=str,
        default="slow",
        help="""tokenizer load mode, can be slow or auto, slow mode load fast but run slow, slow mode is good for debug and test,
                        when you want to get best performance, try auto mode""",
    )
    parser.add_argument(
        "--load_way",
        type=str,
        default="HF",
        help="the way of loading model weights, the default is HF(Huggingface format), llama also supports DS(Deepspeed)",
    )
    parser.add_argument(
        "--max_total_token_num",
        type=int,
        default=6000,
        help="the total token nums the gpu and model can support, equals = max_batch * (input_len + output_len)",
    )
    parser.add_argument(
        "--batch_max_tokens",
        type=int,
        default=None,
        help="max tokens num for new cat batch, it control prefill batch size to Preventing OOM",
    )
    parser.add_argument("--eos_id", type=int, default=2, help="eos stop token id")
    parser.add_argument(
        "--running_max_req_size",
        type=int,
        default=1000,
        help="the max size for forward requests in the same time",
    )
    parser.add_argument(
        "--tp", type=int, default=1, help="model tp parral size, the default is 1"
    )
    parser.add_argument(
        "--max_req_input_len",
        type=int,
        default=None,
        help="the max value for req input tokens num. If None, it will be derived from the config.",
    )
    parser.add_argument(
        "--max_req_total_len",
        type=int,
        default=None,
        help="the max value for req_input_len + req_output_len. If None, it will be derived from the config.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=[],
        nargs="+",
        help="""Model mode: [triton_int8kv | ppl_int8kv | ppl_fp16 | triton_flashdecoding
                        | triton_gqa_attention | triton_gqa_flashdecoding]
                        [triton_int8weight | triton_int4weight | lmdeploy_int4weight | ppl_int4weight],
                        triton_flashdecoding mode is for long context, current support llama llama2 qwen;
                        triton_gqa_attention and triton_gqa_flashdecoding is fast kernel for model which use GQA;
                        triton_int8kv mode use int8 to store kv cache, can increase token capacity, use triton kernel;
                        ppl_int8kv mode use int8 to store kv cache, and use ppl fast kernel;
                        ppl_fp16 mode use ppl fast fp16 decode attention kernel;
                        triton_int8weight and triton_int4weight and lmdeploy_int4weight or ppl_int4weight mode use int8 and int4 to store weights;
                        you need to read source code to make sure the supported detail mode for all models""",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether or not to allow for custom models defined on the Hub in their own modeling files.",
    )
    parser.add_argument(
        "--disable_log_stats",
        action="store_true",
        help="disable logging throughput stats.",
    )
    parser.add_argument(
        "--log_stats_interval",
        type=int,
        default=10,
        help="log stats interval in second.",
    )

    parser.add_argument(
        "--router_token_ratio",
        type=float,
        default=0.0,
        help="token ratio to control router dispatch",
    )
    parser.add_argument(
        "--router_max_new_token_len",
        type=int,
        default=1024,
        help="the request max new token len for router",
    )

    parser.add_argument(
        "--no_skipping_special_tokens",
        action="store_true",
        help="whether to skip special tokens when decoding",
    )
    parser.add_argument(
        "--no_spaces_between_special_tokens",
        action="store_true",
        help="whether to add spaces between special tokens when decoding",
    )

    parser.add_argument(
        "--splitfuse_mode", action="store_true", help="use splitfuse mode"
    )
    parser.add_argument(
        "--splitfuse_block_size", type=int, default=256, help="splitfuse block size"
    )
    parser.add_argument(
        "--prompt_cache_strs",
        type=str,
        default=[],
        nargs="+",
        help="""prompt cache strs""",
    )
    parser.add_argument(
        "--cache_capacity",
        type=int,
        default=200,
        help="cache server capacity for multimodal resources",
    )
    parser.add_argument(
        "--cache_reserved_ratio",
        type=float,
        default=0.5,
        help="cache server reserved capacity ratio after clear",
    )
    parser.add_argument(
        "--return_all_prompt_logprobs",
        action="store_true",
        help="return all prompt tokens logprobs",
    )
    parser.add_argument(
        "--long_truncation_mode",
        type=str,
        choices=[None, "head", "center"],
        default=None,
        help="""use to select the handle way when input token len > max_req_input_len.
                        None : raise Exception
                        head : remove some head tokens to make input token len <= max_req_input_len
                        center : remove some tokens in center loc to make input token len <= max_req_input_len""",
    )

    args = parser.parse_args()

    # 非splitfuse 模式，不支持 prompt cache 特性
    if not args.splitfuse_mode:
        assert len(args.prompt_cache_strs) == 0

    model_config = AutoConfig.from_pretrained(args.model_dir)
    context_length = get_context_length(model_config)

    if args.max_req_input_len is None:
        args.max_req_input_len = context_length - 1
    if args.max_req_total_len is None:
        args.max_req_total_len = context_length

    assert args.max_req_input_len < args.max_req_total_len
    assert args.max_req_total_len <= args.max_total_token_num

    if not args.splitfuse_mode:
        # 普通模式下
        if args.batch_max_tokens is None:
            batch_max_tokens = int(1 / 6 * args.max_total_token_num)
            batch_max_tokens = max(batch_max_tokens, args.max_req_total_len)
            args.batch_max_tokens = batch_max_tokens
        else:
            assert (
                args.batch_max_tokens >= args.max_req_total_len
            ), "batch_max_tokens must >= max_req_total_len"
    else:
        # splitfuse 模式下
        # assert args.batch_max_tokens is not None, "need to set by yourself"
        if args.batch_max_tokens is None:
            batch_max_tokens = int(1 / 6 * args.max_total_token_num)
            batch_max_tokens = max(batch_max_tokens, args.splitfuse_block_size)
            args.batch_max_tokens = batch_max_tokens

    can_use_ports = alloc_can_use_network_port(num=6 + args.tp)

    assert can_use_ports is not None, "Can not alloc enough free ports."
    (
        router_port,
        detokenization_port,
        httpserver_port,
        visual_port,
        cache_port,
        nccl_port,
    ) = can_use_ports[0:6]
    args.nccl_port = nccl_port
    model_rpc_ports = can_use_ports[6:]

    global httpserver_manager
    httpserver_manager = HttpServerManager(
        args,
        router_port=router_port,
        cache_port=cache_port,
        visual_port=visual_port,
        httpserver_port=httpserver_port,
        enable_multimodal=False,
    )

    start_submodule_processes(
        start_funcs=[start_router_process, start_detokenization_process],
        start_args=[
            (args, router_port, detokenization_port, model_rpc_ports),
            (args, detokenization_port, httpserver_port),
        ],
    )
    worker = LightLLMWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_dir,
        args.model_names,
        args.limit_worker_concurrency,
        args.no_register,
        args.conv_template,
        httpserver_manager.tokenizer,
        context_length,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
