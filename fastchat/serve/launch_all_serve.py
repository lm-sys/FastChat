"""
Usage: python launch_all_serve_by_shell.py --model-path-address "THUDM/chatglm2-6b@localhost@2021" "huggyllama/llama-7b@localhost@2022" 

Workers are listed in format of `model-path`@`host`@`port` 

The key mechanism behind this scripts is: 
    1, execute shell cmd to launch the controller/worker/openai-api-server;
    2, check the log of controller/worker/openai-api-server to ensure that the serve is launched properly.
Note that a few of non-critical `fastchat.serve` cmd options are not supported currently.
"""
# NOTE: This is imported and called as soon as possible, before imports that use CUDA.
from fastchat.args.set_args import set_args
args = set_args(["launch_all_serve"])

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import subprocess
import re
import argparse

LOGDIR = "./logs/"

if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)

controller_args = ["controller-host", "controller-port", "dispatch-method"]

worker_args = [
    "worker-host",
    "worker-port",
    "model-path",
    "revision",
    "device",
    "gpus",
    "num-gpus",
    "max-gpu-memory",
    "load-8bit",
    "cpu-offloading",
    "gptq-ckpt",
    "gptq-wbits",
    "gptq-groupsize",
    "gptq-act-order",
    "model-names",
    "limit-worker-concurrency",
    "stream-interval",
    "no-register",
    "controller-address",
]

# openai server
server_args = [
    "server-host",
    "server-port",
    "allow-credentials",
    "api-keys",
    "controller-address",
]

# 0,controller, model_worker, openai_api_server
# 1, cmd options
# 2,LOGDIR
# 3, log file name
base_launch_sh = "nohup python3 -m fastchat.serve.{0} {1} >{2}/{3}.log 2>&1 &"

# 0 LOGDIR
#! 1 log file name
# 2 controller, worker, openai_api_server
base_check_sh = """while [ `grep -c "Uvicorn running on" {0}/{1}.log` -eq '0' ];do
                        sleep 1s;
                        echo "wait {2} running"
                done
                echo '{2} running' """


def string_args(args, args_list):
    args_str = ""
    for key, value in args._get_kwargs():
        key = key.replace("_", "-")
        if key not in args_list:
            continue

        key = key.split("-")[-1] if re.search("port|host", key) else key
        if not value:
            pass
        # 1==True ->  True
        elif isinstance(value, bool) and value == True:
            args_str += f" --{key} "
        elif (
            isinstance(value, list)
            or isinstance(value, tuple)
            or isinstance(value, set)
        ):
            value = " ".join(value)
            args_str += f" --{key} {value} "
        else:
            args_str += f" --{key} {value} "

    return args_str


def launch_worker(item):
    log_name = (
        item.split("/")[-1]
        .split("\\")[-1]
        .replace("-", "_")
        .replace("@", "_")
        .replace(".", "_")
    )

    args.model_path, args.worker_host, args.worker_port = item.split("@")
    print("*" * 80)
    worker_str_args = string_args(args, worker_args)
    print(worker_str_args)
    worker_sh = base_launch_sh.format(
        "model_worker", worker_str_args, LOGDIR, f"worker_{log_name}"
    )
    worker_check_sh = base_check_sh.format(LOGDIR, f"worker_{log_name}", "model_worker")
    subprocess.run(worker_sh, shell=True, check=True)
    subprocess.run(worker_check_sh, shell=True, check=True)


def launch_all():
    controller_str_args = string_args(args, controller_args)
    controller_sh = base_launch_sh.format(
        "controller", controller_str_args, LOGDIR, "controller"
    )
    controller_check_sh = base_check_sh.format(LOGDIR, "controller", "controller")
    subprocess.run(controller_sh, shell=True, check=True)
    subprocess.run(controller_check_sh, shell=True, check=True)

    if isinstance(args.model_path_address, str):
        launch_worker(args.model_path_address)
    else:
        for idx, item in enumerate(args.model_path_address):
            print(f"loading {idx}th model:{item}")
            launch_worker(item)

    server_str_args = string_args(args, server_args)
    server_sh = base_launch_sh.format(
        "openai_api_server", server_str_args, LOGDIR, "openai_api_server"
    )
    server_check_sh = base_check_sh.format(
        LOGDIR, "openai_api_server", "openai_api_server"
    )
    subprocess.run(server_sh, shell=True, check=True)
    subprocess.run(server_check_sh, shell=True, check=True)


if __name__ == "__main__":
    launch_all()
