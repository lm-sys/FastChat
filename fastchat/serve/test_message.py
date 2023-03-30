import argparse
import json

import requests

from fastchat.conversation import default_conversation


def main():
    if args.worker_address:
        worker_addr = args.worker_address
    else:
        controller_addr = args.controller_address
        ret = requests.post(controller_addr + "/refresh_all_workers")
        ret = requests.post(controller_addr + "/list_models")
        models = ret.json()["models"]
        models.sort()
        print(f"Models: {models}")

        ret = requests.post(controller_addr + "/get_worker_address",
            json={"model": args.model_name})
        worker_addr = ret.json()["address"]
        print(f"worker_addr: {worker_addr}")

    if worker_addr == "":
        return

    conv = default_conversation.copy()
    conv.append_message(conv.roles[0], args.message)
    prompt = conv.get_prompt()

    headers = {"User-Agent": "fastchat Client"}
    pload = {
        "model": args.model_name,
        "prompt": prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.7,
        "stop": conv.sep,
    }
    response = requests.post(worker_addr + "/worker_generate_stream", headers=headers,
            json=pload, stream=True)

    print(prompt.replace(conv.sep, "\n"), end="")
    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"].split(conv.sep)[-1]
            print(output, end="\r")
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-address", type=str, default="http://localhost:21001")
    parser.add_argument("--worker-address", type=str)
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--message", type=str, default=
        "Tell me a story with more than 1000 words.")
    args = parser.parse_args()

    main()
