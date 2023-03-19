import argparse
import json

import requests


def main():
    controller_addr = args.url
    ret = requests.post(controller_addr + "/get_worker_address",
            json={"model_name": args.model_name})
    worker_addr = ret.json()["address"]
    print(f"worker_addr: {worker_addr}")

    if worker_addr == "":
        return

    stop_str = "###"
    context = (
        f"A chat between a curious human and a knowledgeable artificial intelligence assistant.{stop_str}"
        f"Human: Hello! What can you do?{stop_str}"
        f"Assistant: As an AI assistant, I can answer questions and chat with you.{stop_str}"
        f"Human: What is the name of the tallest mountain in the world?{stop_str}"
        f"Assistant: Everest.{stop_str}"
        f"Human: What is the weather today?{stop_str}"
        f"Assistant:"
    )

    headers = {"User-Agent": "Alpa Client"}
    pload = {
        "model": "facebook/opt-350m",
        "prompt": context,
        "max_new_tokens": 64,
        "temperature": 0.8,
        "stop": stop_str,
    }
    response = requests.post(worker_addr + "/generate_stream", headers=headers,
            json=pload, stream=True)

    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"].split(f"{stop_str}Assistant: ")[-1]
            print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:21001")
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    args = parser.parse_args()

    main()
