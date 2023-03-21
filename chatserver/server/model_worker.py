import argparse
import dataclasses
import logging
import json
import time
from typing import List, Union
import threading
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn

from chatserver.constants import WORKER_HEART_BEAT_INTERVAL
from chatserver.utils import build_logger

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def heart_beat_worker(controller):

    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


def load_model(model_name, num_gpus):
    disable_torch_init()

    if num_gpus == 1:
        kwargs = {}
    else:
        kwargs = {
            "device_map": "auto",
            "max_memory": {i: "13GiB" for i in range(num_gpus)},
        }

    if model_name == "facebook/llama-7b":
        from transformers import LlamaForCausalLM, LlamaTokenizer
        hf_model_name = "/home/ubuntu/llama_weights/hf-llama-7b/"
        tokenizer = AutoTokenizer.from_pretrained(
           hf_model_name + "tokenizer/")
        model = AutoModelForCausalLM.from_pretrained(
           hf_model_name + "llama-7b/", torch_dtype=torch.float16, **kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
           model_name, torch_dtype=torch.float16, **kwargs)

    if num_gpus == 1:
        model.cuda()

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len


class ModelWorker:
    def __init__(self, controller_addr, worker_addr,
                 worker_id, model_name, num_gpus):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.model_name = model_name

        logger.info(f"Loading the model {model_name} on worker {worker_id} ...")
        self.tokenizer, self.model, self.context_len = load_model(model_name, num_gpus)

        self.register_to_controller()
        self.heart_beat_thread = threading.Thread(
            target=heart_beat_worker, args=(self,))
        self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("register to controller")

        url = self.controller_addr + "/register_model_worker"
        data = {
            "model_name": self.model_name,
            "worker_name": self.worker_addr,
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        url = self.controller_addr + "/receive_heart_beat"
        ret = requests.post(url, json={
            "worker_name": self.worker_addr})
        assert ret.status_code == 200
        exist = ret.json()["exist"]
        if not exist:
            self.register_to_controller()

    @torch.inference_mode()
    def generate_stream(self, args):
        #cur_mem = torch.cuda.memory_allocated()
        #max_mem = torch.cuda.max_memory_allocated()
        #logging.info(f"cur mem: {cur_mem/GB:.2f} GB, max_mem: {max_mem/GB:.2f} GB")

        tokenizer, model = self.tokenizer, self.model

        context = args["prompt"]
        max_new_tokens = args.get("max_new_tokens", 256)
        stop_str = args.get("stop", None)
        temperature = float(args.get("temperature", 1.0))

        input_ids = tokenizer(context).input_ids
        output_ids = list(input_ids)

        max_src_len = self.context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]

        for i in range(max_new_tokens):
            if i == 0:
                out = model(
                    torch.as_tensor([input_ids]).cuda(), use_cache=True)
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                attention_mask = torch.ones(1, past_key_values[0][0].shape[-2] + 1).cuda()
                out = model(input_ids=torch.as_tensor([[token]]).cuda(),
                            use_cache=True,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values)
                logits = out.logits
                past_key_values = out.past_key_values

            assert out.hidden_states is None
            assert out.attentions is None

            last_token_logits = logits[0][-1]
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))
            if token == tokenizer.eos_token_id:
                break

            output_ids.append(token)
            output = tokenizer.decode(output_ids, skip_special_tokens=True)

            if output.endswith(stop_str):
                output = output[:-len(stop_str)]
                stopped = True
            else:
                stopped = False

            ret = {
                "text": output,
                "error": 0,
            }
            yield (json.dumps(ret) + "\0").encode("utf-8")

            if stopped:
                break

        del past_key_values


app = FastAPI()


@app.post("/generate_stream")
async def generate_stream(request: Request):
    args = await request.json()
    return StreamingResponse(worker.generate_stream(args))


@app.post("/check_status")
async def check_status(request: Request):
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str,
        default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:21001")
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--num-gpus", type=int, default=1)
    args = parser.parse_args()

    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         worker_id,
                         args.model_name,
                         args.num_gpus)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
