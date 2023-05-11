"""
A model worker executes the model based on Cacheflow.

Install Cacheflow first. Then, assuming controller is live:
1. ray start --head
2. python3 -m fastchat.serve.cacheflow_worker --model-path path_to_vicuna

launch Gradio:
3. python3 -m fastchat.serve.gradio_web_server --concurrency-count 10000
"""
import argparse
import asyncio
import json
import threading
import time
import uuid
from typing import List, Dict

import requests
import torch
import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer

from cacheflow.master.server import Server, initialize_ray_cluster
from cacheflow.sampling_params import SamplingParams
from cacheflow.sequence import Sequence, SequenceGroup
from cacheflow.utils import Counter, get_gpu_memory, get_cpu_memory
from fastchat.constants import WORKER_HEART_BEAT_INTERVAL
from fastchat.utils import build_logger, pretty_print_semaphore

GB = 1 << 30
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")
global_counter = 0
seed = torch.cuda.current_device()


def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class CacheFlowWorker:
    def __init__(
        self,
        controller_addr,
        worker_addr,
        worker_id,
        no_register,
        model_path,
        model_name,
        block_size,
        seed,
        swap_space,
        max_num_batched_tokens,
        distributed_init_method,
        all_stage_devices,
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        self.model_name = model_name or model_path.split("/")[-1]

        logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")
        self.block_size = block_size

        # FIXME(Hao): we need to pass the tokenizer into cacheflow because we need
        # to detect the stopping criteria "###".
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.seq_group_counter = Counter()
        self.seq_counter = Counter()
        # FIXME(Hao): hard code context len
        self.context_len = 2048
        # pipeline_parallel_size = 1,
        # tensor_parallel_size = 1,
        # dtype = torch.float16
        remote_server_class = Server
        self.server = remote_server_class(
            model=self.model_name,
            model_path=model_path,
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            block_size=block_size,
            dtype=torch.float16,
            seed=seed,
            swap_space=swap_space,
            max_num_batched_tokens=max_num_batched_tokens,
            num_nodes=1,
            num_devices_per_node=4,
            distributed_init_method=distributed_init_method,
            all_stage_devices=all_stage_devices,
            gpu_memory=get_gpu_memory(),
            cpu_memory=get_cpu_memory(),
        )
        self.running_seq_groups: Dict[int, SequenceGroup] = {}
        self.sequence_group_events: Dict[int, asyncio.Event] = {}
        self.is_server_running = False

        if not no_register:
            time.sleep(30)  # wait for model loading
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,)
            )
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status(),
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(
            f"Send heart beat. Models: {[self.model_name]}. "
            f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
            f"global_counter: {global_counter}"
        )

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(
                    url,
                    json={
                        "worker_name": self.worker_addr,
                        "queue_length": self.get_queue_length(),
                    },
                    timeout=5,
                )
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if (
            model_semaphore is None
            or model_semaphore._value is None
            or model_semaphore._waiters is None
        ):
            return 0
        else:
            return (
                args.limit_model_concurrency
                - model_semaphore._value
                + len(model_semaphore._waiters)
            )

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    async def server_step(self):
        self.is_server_running = True
        updated_seq_groups = self.server.step()
        self.is_server_running = False
        # Notify the waiting coroutines that there new outputs ready.
        for seq_group in updated_seq_groups:
            group_id = seq_group.group_id
            self.running_seq_groups[group_id] = seq_group
            self.sequence_group_events[group_id].set()

    async def generate_stream(self, params):
        tokenizer = self.tokenizer
        context = params["prompt"]
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)
        echo = params.get("echo", True)
        stop_token_ids = params.get("stop_token_ids", None) or []
        stop_token_ids.append(tokenizer.eos_token_id)

        input_ids = tokenizer(context).input_ids
        max_src_len = self.context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]

        # make sampling params in cacheflow
        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            top_p = 1.0
        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            use_beam_search=False,
            stop_token_ids=stop_token_ids,
            max_num_steps=max_new_tokens,
            num_logprobs=0,
            context_window_size=None,
        )

        if stop_str is not None:
            sampling_params.stop_str = stop_str
        # we might sample multiple sequences, but in chatbot, this is one
        seqs: List[Sequence] = []
        for _ in range(sampling_params.n):
            seq_id = next(self.seq_counter)
            seq = Sequence(seq_id, input_ids, block_size=self.block_size)
            seqs.append(seq)

        arrival_time = time.time()
        group_id = next(self.seq_group_counter)
        # logger.info(f"Group {group_id} arrives at {time.time()}")
        seq_group = SequenceGroup(group_id, seqs, arrival_time)
        group_event = asyncio.Event()
        self.running_seq_groups[group_id] = seq_group
        self.sequence_group_events[group_id] = group_event
        self.server.add_sequence_groups([(seq_group, sampling_params)])
        while True:
            if not self.is_server_running:
                await self.server_step()
            try:
                await asyncio.wait_for(
                    group_event.wait(), timeout=TIMEOUT_TO_PREVENT_DEADLOCK
                )
            except:
                pass
            group_event.clear()
            seq_group = self.running_seq_groups[group_id]
            all_outputs = []
            for seq in seq_group.seqs:
                token_ids = seq.get_token_ids()
                if not echo:
                    token_ids = token_ids[len(input_ids) :]
                output = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                if stop_str is not None:
                    if output.endswith(stop_str):
                        output = output[: -len(stop_str)]
                all_outputs.append(output)
            assert len(seq_group.seqs) == 1
            ret = {
                "text": all_outputs[0],
                "error_code": 0,
            }
            yield (json.dumps(ret) + "\0").encode("utf-8")
            if seq_group.is_finished():
                del self.running_seq_groups[group_id]
                del self.sequence_group_events[group_id]
                break


app = FastAPI()
model_semaphore = None


def release_model_semaphore():
    model_semaphore.release()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_model_semaphore)
    # return StreamingResponse(generator, background=background_tasks)
    return StreamingResponse(
        worker.generate_stream(params), background=background_tasks
    )


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument(
        "--model-path", type=str, default="/home/haozhang/weights/hf-llama-7b"
    )
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--limit-model-concurrency", type=int, default=1024)
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    # cacheflow specific params
    parser.add_argument(
        "--block-size", type=int, default=8, choices=[8, 16], help="token block size"
    )
    parser.add_argument(
        "--swap-space", type=int, default=20, help="CPU swap space size (GiB) per GPU"
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=2560,
        help="maximum number of batched tokens",
    )
    args = parser.parse_args()

    (
        num_nodes,
        num_devices_per_node,
        distributed_init_method,
        all_stage_devices,
    ) = initialize_ray_cluster(pipeline_parallel_size=1, tensor_parallel_size=1)

    worker = CacheFlowWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.no_register,
        args.model_path,
        args.model_name,
        args.block_size,
        seed,
        args.swap_space,
        args.max_num_batched_tokens,
        distributed_init_method,
        all_stage_devices,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
