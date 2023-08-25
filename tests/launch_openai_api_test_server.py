"""
Launch an OpenAI API server with multiple model workers.
"""
import os


def launch_process(cmd):
    os.popen(cmd)


if __name__ == "__main__":
    launch_process("python3 -m fastchat.serve.controller")
    launch_process("python3 -m fastchat.serve.openai_api_server")

    models = [
        "lmsys/vicuna-7b-v1.3",
        "lmsys/fastchat-t5-3b-v1.0",
        "THUDM/chatglm-6b",
        "mosaicml/mpt-7b-chat",
    ]

    for i, model_path in enumerate(models):
        launch_process(
            f"CUDA_VISIBLE_DEVICES={i} python3 -m fastchat.serve.model_worker "
            f"--model-path {model_path} --port {30000+i} --worker http://localhost:{30000+i}"
        )

    while True:
        pass
