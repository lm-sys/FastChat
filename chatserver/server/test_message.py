import argparse
import json

import requests

from chatserver.conversation import default_conversation


def main():
    controller_addr = args.url
    ret = requests.post(controller_addr + "/get_worker_address",
            json={"model_name": args.model_name})
    worker_addr = ret.json()["address"]
    print(f"worker_addr: {worker_addr}")

    if worker_addr == "":
        return

    conv = default_conversation.copy()
    conv.append_message(conv.roles[0], "What is the weather today?")
    prompt = conv.get_prompt()

    headers = {"User-Agent": "Alpa Client"}
    pload = {
        "model": "facebook/opt-350m",
        "prompt": prompt,
        "max_new_tokens": 32,
        "temperature": 0.8,
        "stop": conv.sep,
    }
    response = requests.post(worker_addr + "/generate_stream", headers=headers,
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
    parser.add_argument("--url", type=str, default="http://localhost:21001")
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    args = parser.parse_args()

    main()
