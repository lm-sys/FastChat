import argparse
import dataclasses
import logging
import json
import time
from typing import List, Union
import threading

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import requests
from transformers import AutoTokenizer, OPTForCausalLM
import torch
import uvicorn

from chatserver.server.constants import WORKER_HEART_BEAT_INTERVAL


logger = logging.getLogger("model_worker")


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


class ModelWorker:
    def __init__(self, controller_addr, worker_addr, model_name):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.model_name = model_name

        disable_torch_init()
        self.tokenizer = AutoTokenizer.from_pretrained(
           model_name, add_bos_token=False)
        self.model = OPTForCausalLM.from_pretrained(
           model_name, torch_dtype=torch.float16).cuda()

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

    def generate_stream(self, args):
        tokenizer, model = self.tokenizer, self.model

        context = args["prompt"]
        max_new_tokens = args.get("max_new_tokens", 1024)
        stop_str = args.get("stop", None)
        temperature = float(args.get("temperature", 1.0))

        if stop_str:
            if tokenizer.add_bos_token:
                assert len(tokenizer(stop_str).input_ids) == 2
                stop_token = tokenizer(stop_str).input_ids[1]
            else:
                assert len(tokenizer(stop_str).input_ids) == 1
                stop_token = tokenizer(stop_str).input_ids[0]
        else:
            stop_token = None

        input_ids = tokenizer(context).input_ids
        output_ids = list(input_ids)

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

            last_token_logits = logits[0][-1]
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))
            if token == stop_token:
                break

            output_ids.append(token)
            output = tokenizer.decode(output_ids, skip_special_tokens=True)

            ret = {
                "text": output,
                "error": 0,
            }
            yield (json.dumps(ret) + "\0").encode("utf-8")



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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         args.model_name)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
