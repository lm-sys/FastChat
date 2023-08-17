import argparse

from fastchat.args.add_model_args import add_model_args
from fastchat.args.add_cli_args import add_cli_args
from fastchat.args.add_huggingface_args import add_huggingface_args
from fastchat.args.add_model_worker_args import add_model_worker_args
from fastchat.args.add_multi_model_worker_args import add_multi_model_worker_args
from fastchat.args.add_launch_all_serve_args import add_launch_all_serve_args
from fastchat.args.set_devices import set_devices


def set_args(add_list):
    parser = argparse.ArgumentParser()
    if "model" in add_list:
        add_model_args(parser)
    if "cli" in add_list:
        add_cli_args(parser)
    if "huggingface" in add_list:
        add_huggingface_args(parser)
    if "model_worker" in add_list:
        add_model_worker_args(parser)
    if "multi_model_worker" in add_list:
        add_multi_model_worker_args(parser)
    if "launch_all_serve" in add_list:
        add_launch_all_serve_args(parser)
    args = parser.parse_args()
    if "huggingface" in add_list:
        # Reset default repetition penalty for T5 models.
        if "t5" in args.model_path and args.repetition_penalty == 1.0:
            args.repetition_penalty = 1.2
    if "launch_all_serve" in add_list:
        set_devices(args)
        return argparse.Namespace(
            **vars(args),
            **{"controller-address": f"http://{args.controller_host}:{args.controller_port}"},
        )
    set_devices(args)
    return args