"""
Usage: python launch_all_serve_by_shell.py --model-path-address "THUDM/chatglm2-6b@localhost@2021" "huggyllama/llama-7b@localhost@2022" 

Workers are listed in format of `model-path`@`host`@`port` 

The key mechanism behind this scripts is: 
    1, execute shell cmd to launch the controller/worker/openai-api-server;
    2, check the log of controller/worker/openai-api-server to ensure that the serve is launched properly.
Note that a few of non-critical `fastchat.serve` cmd options are not supported currently.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import subprocess
import re
import argparse

LOGDIR = "./logs/"

if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)

parser = argparse.ArgumentParser()
# ------multi worker-----------------
parser.add_argument(
    "--model-path-address",
    default="THUDM/chatglm2-6b@localhost@20002",
    nargs="+",
    type=str,
    help="model path, host, and port, formatted as model-path@host@port",
)
# ---------------controller-------------------------

parser.add_argument("--controller-host", type=str, default="localhost")
parser.add_argument("--controller-port", type=int, default=21001)
parser.add_argument(
    "--dispatch-method",
    type=str,
    choices=["lottery", "shortest_queue"],
    default="shortest_queue",
)
controller_args = ["controller-host", "controller-port", "dispatch-method"]

# ----------------------worker------------------------------------------

parser.add_argument("--worker-host", type=str, default="localhost")
parser.add_argument("--worker-port", type=int, default=21002)
# parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
# parser.add_argument(
#     "--controller-address", type=str, default="http://localhost:21001"
# )
parser.add_argument(
    "--model-path",
    type=str,
    default="lmsys/vicuna-7b-v1.5",
    help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
)
parser.add_argument(
    "--revision",
    type=str,
    default="main",
    help="Hugging Face Hub model revision identifier",
)
parser.add_argument(
    "--device",
    type=str,
    choices=["cpu", "cuda", "mps", "xpu", "npu"],
    default="cuda",
    help="The device type",
)
parser.add_argument(
    "--gpus",
    type=str,
    default="0",
    help="A single GPU like 1 or multiple GPUs like 0,2",
)
parser.add_argument("--num-gpus", type=int, default=1)
parser.add_argument(
    "--max-gpu-memory",
    type=str,
    help="The maximum memory per gpu. Use a string like '13Gib'",
)
parser.add_argument("--load-8bit", action="store_true", help="Use 8-bit quantization")
parser.add_argument(
    "--cpu-offloading",
    action="store_true",
    help="Only when using 8-bit quantization: Offload excess weights to the CPU that don't fit on the GPU",
)
parser.add_argument(
    "--gptq-ckpt",
    type=str,
    default=None,
    help="Load quantized model. The path to the local GPTQ checkpoint.",
)
parser.add_argument(
    "--gptq-wbits",
    type=int,
    default=16,
    choices=[2, 3, 4, 8, 16],
    help="#bits to use for quantization",
)
parser.add_argument(
    "--gptq-groupsize",
    type=int,
    default=-1,
    help="Groupsize to use for quantization; default uses full row.",
)
parser.add_argument(
    "--gptq-act-order",
    action="store_true",
    help="Whether to apply the activation order GPTQ heuristic",
)
parser.add_argument(
    "--model-names",
    type=lambda s: s.split(","),
    help="Optional display comma separated names",
)
parser.add_argument(
    "--limit-worker-concurrency",
    type=int,
    default=5,
    help="Limit the model concurrency to prevent OOM.",
)
parser.add_argument("--stream-interval", type=int, default=2)
parser.add_argument("--no-register", action="store_true")

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
# -----------------openai server---------------------------

parser.add_argument("--server-host", type=str, default="localhost", help="host name")
parser.add_argument("--server-port", type=int, default=8001, help="port number")
parser.add_argument(
    "--allow-credentials", action="store_true", help="allow credentials"
)
# parser.add_argument(
#     "--allowed-origins", type=json.loads, default=["*"], help="allowed origins"
# )
# parser.add_argument(
#     "--allowed-methods", type=json.loads, default=["*"], help="allowed methods"
# )
# parser.add_argument(
#     "--allowed-headers", type=json.loads, default=["*"], help="allowed headers"
# )
parser.add_argument(
    "--api-keys",
    type=lambda s: s.split(","),
    help="Optional list of comma separated API keys",
)
server_args = [
    "server-host",
    "server-port",
    "allow-credentials",
    "api-keys",
    "controller-address",
]

args = parser.parse_args()

args = argparse.Namespace(
    **vars(args),
    **{"controller-address": f"http://{args.controller_host}:{args.controller_port}"},
)

if args.gpus:
    if len(args.gpus.split(",")) < args.num_gpus:
        raise ValueError(
            f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

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
