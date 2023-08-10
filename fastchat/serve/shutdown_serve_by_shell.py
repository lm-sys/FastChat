"""
Usage：
python llm_api_shutdown.py --serve all
options: "all","controller","model_worker","openai_api_server"， `all` means to stop all related serves 
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--serve", choices=["all", "controller", "model_worker", "openai_api_server"]
)

args = parser.parse_args()

base_shell = "ps -eo user,pid,cmd|grep fastchat.serve{}|grep -v grep|awk '{{print $2}}'|xargs kill -9"

if args.serve == "all":
    shell_script = base_shell.format("")

else:
    serve = f".{args.serve}"
    shell_script = base_shell.format(serve)
print(f"execute shell cmd: {shell_script}")
subprocess.run(shell_script, shell=True, check=True)
print(f"llm api sever --{args.serve} has been shutdown!")
