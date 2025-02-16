"""
Usage：
python shutdown_serve.py --down all
options: "all","controller","model_worker","openai_api_server"， `all` means to stop all related servers 
"""

import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument(
    "--down", choices=["all", "controller", "model_worker", "openai_api_server"]
)
args = parser.parse_args()
base_shell = "ps -eo user,pid,cmd|grep fastchat.serve{}|grep -v grep|awk '{{print $2}}'|xargs kill -9"
if args.down == "all":
    shell_script = base_shell.format("")
else:
    serve = f".{args.down}"
    shell_script = base_shell.format(serve)
print(f"execute shell cmd: {shell_script}")
subprocess.run(shell_script, shell=True, check=True)
print(f"{args.down} has been shutdown!")
