import argparse
import dataclasses
import logging
from typing import List, Union

from fastapi import FastAPI
import requests
import uvicorn

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class ModelInfo:
    worker_addresses: List[str]
    worker_pt: int


def check_liveness(address):
    x = requests.get(address + "/status", timeout=30)
    return x.status_code == 200


class Controller:
    def __init__(self):
        logger.info("init controller")

        # Dict[str -> ModelInfo]
        self.model_info = {}

    def register_model_worker(self, model_name: str, worker_address: str):
        if model_name not in self.model_info:
            self.model_info[model_name] = ModelInfo()

        info = self.model_info[model_name]
        if worker_addresses in info:
            return

        info.worker_addresses.append(model_name)

    def get_worker_address(self, model_name: str):
        if model_name not in self.model_info:
            return ""

        info = self.model_info[model_name]
        info.pt = (info.pt + 1) % len(info.worker_addresses)
        return info.worker_addresses[info.pt]


app = FastAPI()


@app.post("/get_worker_address/")
async def get_worker_address(model_name: str):
    return controller.get_worker_address(model_name)


@app.post("/register_model_worker/")
async def get_worker_address(model_name: str, worker_address: str):
    controller.register_model_worker(model_name, worker_address)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=10001)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    app = FastAPI()
    controller = Controller()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
