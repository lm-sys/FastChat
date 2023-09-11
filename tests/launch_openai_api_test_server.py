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
        ("lmsys/vicuna-7b-v1.5", "model_worker"),
        ("lmsys/fastchat-t5-3b-v1.0", "model_worker"),
        ("THUDM/chatglm-6b", "model_worker"),
        ("mosaicml/mpt-7b-chat", "model_worker"),
        ("meta-llama/Llama-2-7b-chat-hf", "vllm_worker"),
    ]

    for i, (model_path, worker_name) in enumerate(models):
        cmd = (
            f"CUDA_VISIBLE_DEVICES={i} python3 -m fastchat.serve.{worker_name} "
            f"--model-path {model_path} --port {30000+i} "
            f"--worker-address http://localhost:{30000+i} "
        )
        if worker_name == "vllm_worker":
            cmd += "--tokenizer hf-internal-testing/llama-tokenizer"

        launch_process(cmd)

    while True:
        pass
