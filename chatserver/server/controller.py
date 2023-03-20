import argparse
import dataclasses
import logging
import time
from typing import List, Union
import threading

from fastapi import FastAPI, Request
import requests
import uvicorn

from chatserver.server.constants import CONTROLLER_HEART_BEAT_EXPIRATION

logger = logging.getLogger("controller")


@dataclasses.dataclass
class ModelInfo:
    worker_names: List[str]
    worker_pt: int


@dataclasses.dataclass
class WorkerInfo:
    model_names: List[str]
    last_heart_beat: str


def heart_beat_controller(controller):

    while True:
        time.sleep(CONTROLLER_HEART_BEAT_EXPIRATION)
        controller.remove_stable_workers_by_expiration()


class Controller:
    def __init__(self):
        # Dict[str -> ModelInfo]
        self.model_info = {}

        # Dict[str -> WorkerInfo]
        self.worker_info = {}

        self.heart_beat_thread = threading.Thread(
            target=heart_beat_controller, args=(self,))
        self.heart_beat_thread.start()

        logger.info("init controller")

    def register_model_worker(self, model_name: str, worker_name: str):
        if model_name not in self.model_info:
            self.model_info[model_name] = ModelInfo([], 0)

        if worker_name not in self.worker_info:
            self.worker_info[worker_name] = WorkerInfo([], 0)

        m_info = self.model_info[model_name]
        w_info = self.worker_info[worker_name]

        if worker_name in m_info.worker_names:
            logger.info(f"Register existing. {(model_name, worker_name)}")
            return
        assert model_name not in w_info.model_names

        m_info.worker_names.append(worker_name)
        w_info.model_names.append(model_name)
        w_info.last_heart_beat = time.time()

        logger.info(f"Register new. {(model_name, worker_name)}")

        self.remove_stable_workers_by_checking()

    def get_worker_address(self, model_name: str):
        if model_name not in self.model_info:
            return ""

        info = self.model_info[model_name]

        while True:
            if len(info.worker_names) == 0:
                return ""
            info.worker_pt = (info.worker_pt + 1) % len(info.worker_names)
            worker_name = info.worker_names[info.worker_pt]

            if self.check_worker_status(worker_name):
                break
            else:
                self.remove_worker(worker_name)
                continue

        return worker_name

    def remove_worker(self, worker_name):
        logger.info(f"Remove worker. {worker_name}")
        for model_name in self.worker_info[worker_name].model_names:
            self.model_info[model_name].worker_names.remove(worker_name)
        del self.worker_info[worker_name]

    def check_worker_status(self, worker_name):
        try:
            r = requests.post(worker_name + "/check_status")
        except requests.exceptions.RequestException:
            return False
        return r.status_code == 200

    def receive_heart_beat(self, worker_name: str):
        if worker_name not in self.worker_info:
            logger.info(f"Receive unknow heart beat. {worker_name}")
            return False

        self.worker_info[worker_name].last_heart_beat = time.time()
        logger.info(f"Receive heart beat. {worker_name}")
        return True

    def remove_stable_workers_by_expiration(self):
        expire = time.time() - CONTROLLER_HEART_BEAT_EXPIRATION
        to_delete = []
        for worker_name, w_info in self.worker_info.items():
            if w_info.last_heart_beat < expire:
                to_delete.append(worker_name)
        
        for worker_name in to_delete:
            self.remove_worker(worker_name)

    def remove_stable_workers_by_checking(self):
        to_delete = []
        for worker_name in self.worker_info:
            if not self.check_worker_status(worker_name):
                to_delete.append(worker_name)

        for worker_name in to_delete:
            self.remove_worker(worker_name)

    def list_models(self):
        models = []
        for model, m_info in self.model_info.items():
            if len(m_info.worker_names) > 0:
                models.append(model)
        return models


app = FastAPI()


@app.post("/get_worker_address")
async def get_worker_address(request: Request):
    data = await request.json()
    addr = controller.get_worker_address(data["model_name"])
    return {"address": addr}


@app.post("/register_model_worker")
async def get_worker_address(request: Request):
    data = await request.json()
    controller.register_model_worker(
        data["model_name"], data["worker_name"])


@app.post("/receive_heart_beat")
async def get_worker_address(request: Request):
    data = await request.json()
    exist = controller.receive_heart_beat(data["worker_name"])
    return {"exist": exist}


@app.post("/list_models")
async def list_models():
    models = controller.list_models()
    return {"models": models}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21001)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    controller = Controller()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
