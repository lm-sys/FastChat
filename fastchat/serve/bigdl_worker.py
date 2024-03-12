"""
A model worker that executes the model based on BigDL-LLM.

See documentations at docs/bigdlllm_integration.md
"""

import argparse
import asyncio
import atexit
import json
from typing import List
import uuid
from threading import Thread
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.serve.model_worker import (
    logger,
    worker_id,
)
from fastchat.serve.base_model_worker import (
    create_background_tasks,
    acquire_worker_semaphore,
    release_worker_semaphore,
)
from fastchat.utils import get_context_length, is_partial_stop

from bigdl.llm.transformers.loader import load_model
from transformers import TextIteratorStreamer

app = FastAPI()


class BigDLLLMWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        conv_template: str = None,
        load_in_low_bit: str = "sym_int4",
        device: str = "cpu",
        no_register: bool = False,
        trust_remote_code: bool = False,
        stream_interval: int = 4,
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

        self.load_in_low_bit = load_in_low_bit
        logger.info(
            f"Loading the model {self.model_names} on worker {worker_id}, worker type: BigDLLLM worker..."
        )

        logger.info(f"Using low bit format: {self.load_in_low_bit}, device: {device}")

        self.device = device

        self.model, self.tokenizer = load_model(
            model_path, device, self.load_in_low_bit, trust_remote_code
        )
        self.stream_interval = stream_interval
        self.context_len = get_context_length(self.model.config)
        if not no_register:
            self.init_heart_beat()

    def generate_stream_gate(self, params):
        self.call_ct += 1
        # context length is self.context_length
        prompt = params["prompt"]
        temperature = float(params.get("temperature", 1.0))
        repetition_penalty = float(params.get("repetition_penalty", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = int(params.get("top_k", -1))  # -1 means disable
        max_new_tokens = int(params.get("max_new_tokens", 256))
        echo = bool(params.get("echo", True))
        stop_str = params.get("stop", None)
        stop_token_ids = params.get("stop_token_ids", None) or []
        if self.tokenizer.eos_token_id not in stop_token_ids:
            stop_token_ids.append(self.tokenizer.eos_token_id)

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

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        if self.device == "xpu":
            input_ids = input_ids.to("xpu")

        input_echo_len = input_ids.shape[1]

        if self.model.config.is_encoder_decoder:
            max_src_len = self.context_len
            input_ids = input_ids[:max_src_len]
            input_echo_len = len(input_ids)
            prompt = self.tokenizer.decode(
                input_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
        else:
            # Truncate the max_new_tokens if input_ids is too long
            new_max_new_tokens = min(self.context_len - input_echo_len, max_new_tokens)
            if new_max_new_tokens < max_new_tokens:
                logger.info(
                    f"Warning: max_new_tokens[{max_new_tokens}] + prompt[{input_echo_len}] greater than context_length[{self.context_len}]"
                )
                logger.info(f"Reset max_new_tokens to {new_max_new_tokens}")
                max_new_tokens = new_max_new_tokens

        # Use TextIteratorStreamer for streaming output
        streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer,
            timeout=60,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Generation config:
        # https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig
        generated_kwargs = dict(
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
        )

        def model_generate():
            self.model.generate(input_ids, **generated_kwargs)

        t1 = Thread(target=model_generate)
        t1.start()

        stopped = False
        finish_reason = None
        if echo:
            partial_output = prompt
            rfind_start = len(prompt)
        else:
            partial_output = ""
            rfind_start = 0

        for i in range(max_new_tokens):
            try:
                output_token = next(streamer)
            except StopIteration:
                # Stop early
                stopped = True
                break
            partial_output += output_token

            if i % self.stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                for each_stop in stop:
                    pos = partial_output.rfind(each_stop, rfind_start)
                if pos != -1:
                    partial_output = partial_output[:pos]
                    stopped = True
                    break
                else:
                    partially_stopped = is_partial_stop(partial_output, each_stop)
                    if partially_stopped:
                        break
                if not partially_stopped:
                    json_output = {
                        "text": partial_output,
                        "usage": {
                            "prompt_tokens": input_echo_len,
                            "completion_tokens": i,
                            "total_tokens": input_echo_len + i,
                        },
                        "finish_reason": None,
                    }
                    ret = {
                        "text": json_output["text"],
                        "error_code": 0,
                    }
                    ret["usage"] = json_output["usage"]
                    ret["finish_reason"] = json_output["finish_reason"]
                    yield json.dumps(ret).encode() + b"\0"

            if stopped:
                break
        else:
            finish_reason = "length"

        if stopped:
            finish_reason = "stop"
        json_output = {
            "text": partial_output,
            "error_code": 0,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": i,
                "total_tokens": input_echo_len + i,
            },
            "finish_reason": finish_reason,
        }
        yield json.dumps(json_output).encode() + b"\0"

    def generate_gate(self, params):
        for x in self.generate_stream_gate(params):
            # for x in self.generate_stream2(params):
            pass
        return json.loads(x[:-1].decode())


# Below are api interfaces
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
    output = await asyncio.to_thread(worker.generate_gate, params)
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
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument("--stream-interval", type=int, default=4)
    parser.add_argument(
        "--low-bit", type=str, default="sym_int4", help="Low bit format."
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for executing model, cpu/xpu"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_false",
        default=True,
        help="Trust remote code (e.g., from HuggingFace) when"
        "downloading the model and tokenizer.",
    )

    args = parser.parse_args()
    worker = BigDLLLMWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        args.conv_template,
        args.low_bit,
        args.device,
        args.no_register,
        args.trust_remote_code,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
