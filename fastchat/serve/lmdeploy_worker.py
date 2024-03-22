"""A model worker that executes the model based on LMDeploy.

See documentations at docs/lmdeploy_integration.md
"""

import argparse
import asyncio
import json
import logging
from typing import List

import uvicorn
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.serve.model_worker import logger, worker_id
from fastchat.utils import is_partial_stop

from lmdeploy import TurbomindEngineConfig, pipeline
from lmdeploy.messages import GenerationConfig
from lmdeploy.serve.async_engine import AsyncEngine

app = FastAPI()


class InterFace:
    request_id: int = 0
    engine: AsyncEngine = None


class LMDeployWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        tokenizer: object,
        session_len: int,
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
            f"Loading the model {self.model_names} on worker {worker_id}, worker type: LMDeploy worker..."
        )
        self.tokenizer = tokenizer
        self.context_len = session_len

        if not no_register:
            self.init_heart_beat()

    async def generate_stream(self, params):
        self.call_ct += 1

        context = params.pop("prompt")
        request_id = params.pop("request_id")
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = params.get("top_k", 1.0)
        repetition_penalty = float(params.get("repetition_penalty", 1.0))
        max_new_tokens = params.get("max_new_tokens", 256)
        stop_str = params.get("stop", None)
        stop_token_ids = params.get("stop_token_ids", None) or []
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.append(self.tokenizer.eos_token_id)
        echo = params.get("echo", True)
        ignore_eos = params.get("ignore_eos", False)
        echo = params.get("echo", True)
        skip_special_tokens = params.get("skip_special_tokens", True)
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

        # make gen_config in lmdeploy
        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            top_p = 1.0

        gen_config = GenerationConfig(
            n=1,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            ignore_eos=ignore_eos,
            stop_words=list(stop),
            repetition_penalty=repetition_penalty,
            skip_special_tokens=skip_special_tokens,
        )
        results_generator = InterFace.engine.generate(
            context, int(request_id), gen_config=gen_config, do_preprocess=False
        )
        text_outputs = ""
        async for request_output in results_generator:
            text_outputs += request_output.response

            partial_stop = any(is_partial_stop(text_outputs, i) for i in stop)
            # prevent yielding partial stop sequence
            if partial_stop:
                continue

            aborted = False
            if request and await request.is_disconnected():
                await InterFace.engine.stop_session(int(request_id))
                request_output.finish_reason = True
                aborted = True

            ret = {
                "text": context + text_outputs if echo else text_outputs,
                "error_code": 0,
                "usage": {
                    "prompt_tokens": request_output.input_token_len,
                    "completion_tokens": request_output.generate_token_len,
                    "total_tokens": request_output.input_token_len
                    + request_output.generate_token_len,
                },
                "finish_reason": request_output.finish_reason,
            }
            if request_output.finish_reason is not None:
                yield (
                    json.dumps({**ret, "finish_reason": None}, ensure_ascii=False)
                    + "\0"
                ).encode("utf-8")
            yield (
                json.dumps(
                    {**ret, "finish_reason": request_output.finish_reason},
                    ensure_ascii=False,
                )
                + "\0"
            ).encode("utf-8")

            if aborted is True:  # In case of abort, we need to break the loop
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
        await InterFace.engine.stop_session(int(request_id))

    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    background_tasks.add_task(abort_request)
    return background_tasks


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = InterFace.request_id
    InterFace.request_id += 1
    params["request_id"] = request_id
    params["request"] = request
    generator = worker.generate_stream(params)
    background_tasks = create_background_tasks(request_id)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = InterFace.request_id
    InterFace.request_id += 1
    params["request_id"] = request_id
    params["request"] = request
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
    parser.add_argument("--model-path", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--limit-worker-concurrency", type=int, default=1024)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
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
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="GPU number used in tensor parallelism. Should be 2^n",
    )
    parser.add_argument(
        "--session-len",
        type=int,
        default=None,
        help="The max session length of a sequence",
    )
    parser.add_argument(
        "--max-batch-size", type=int, default=128, help="Maximum batch size"
    )
    parser.add_argument(
        "--cache-max-entry-count",
        type=float,
        default=0.8,
        help="The percentage of gpu memory occupied by the k/v cache",
    )
    parser.add_argument(
        "--cache-block-seq-len",
        type=int,
        default=64,
        help="The length of the token sequence in a k/v block. "
        "For Turbomind Engine, if the GPU compute capability "
        "is >= 8.0, it should be a multiple of 32, otherwise "
        "it should be a multiple of 64.",
    )
    parser.add_argument(
        "--model-format",
        type=str,
        default=None,
        choices=["hf", "llama", "awq"],
        help="The format of input model. `hf` meaning `hf_llama`, `llama` "
        "meaning `meta_llama`, `awq` meaning the quantized model by awq",
    )
    parser.add_argument(
        "--quant-policy",
        type=int,
        default=0,
        help="Whether to use kv int8. When k/v is " "quantized into 8 bit, set it to 4",
    )
    parser.add_argument(
        "--rope-scaling-factor",
        type=float,
        default=0.0,
        help="Rope scaling factor used for dynamic "
        "ntk, default to 0. TurboMind follows the "
        "implementation of transformer "
        "LlamaAttention",
    )
    parser.add_argument(
        "--use-logn-attn",
        action="store_true",
        default=False,
        help="Whether to use logn attention scaling. default to False",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="ERROR",
        choices=list(logging._nameToLevel.keys()),
        help="Set the log level",
    )
    args = parser.parse_args()

    backend_config = TurbomindEngineConfig(
        model_format=args.model_format,
        tp=args.tp,
        session_len=args.session_len,
        max_batch_size=args.max_batch_size,
        cache_block_seq_len=args.cache_block_seq_len,
        quant_policy=args.quant_policy,
        rope_scaling_factor=args.rope_scaling_factor,
        use_logn_attn=args.use_logn_attn,
    )
    InterFace.engine = pipeline(
        args.model_path, backend_config=backend_config, log_level=args.log_level
    )
    worker = LMDeployWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        args.no_register,
        InterFace.engine.tokenizer.model.model,
        InterFace.engine.session_len,
        args.conv_template,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
