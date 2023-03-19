import dataclasses
from typing import List, Union

from fastapi import FastAPI
import requests


@dataclasses.dataclass
class ModelInfo:
    worker_addresses: List[str]
    worker_pt: int


def check_liveness(address):
    x = requests.get(address + "/status", timeout=30)
    return x.status_code == 200


class Controller:
    def __init__(self):
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
controler = Controller()


@app.post("/generate")
async def get_worker_address():
    return "Generate something"

