"""
Launch an OpenAI API server with lazy-loading multi-model worker.
"""
import os


def launch_process(cmd):
    os.popen(cmd)


if __name__ == "__main__":
    launch_process("python3 -m fastchat.serve.controller")
    launch_process("python3 -m fastchat.serve.openai_api_server")

    models = [
        ("princeton-nlp/Sheared-LLaMA-1.3B", ""),
        ("lmsys/fastchat-t5-3b-v1.0", ""),
        ]

    cmd = ("python3 -m fastchat.serve.multi_model_worker"
           " --lazy --limit-worker-concurrency 1")
    for m in models:
        cmd += " --model-path " + " ".join(m)

    launch_process(cmd)

    while True:
        pass
