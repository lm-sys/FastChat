import argparse
import json

import requests
import threading
import time

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
    # prompt = conv.get_prompt()
    prompt = [f"Tell me a story with more than {''.join([str(i+1)] * 5)} words"
              for i in range(args.n_thread)]

    prompt_len = len(prompt[0].split(" "))
    headers = {"User-Agent": "fastchat Client"}
    ploads = [{
        "model": args.model_name,
        "prompt": prompt[i],
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.0,
        # "stop": conv.sep,
    } for i in range(len(prompt))]

    def send_request(results, i):
        response = requests.post(worker_addr + "/worker_generate_stream", headers=headers,
                                 json=ploads[i], stream=False)
        results[i] = response

    # use args.n_threads to prompt the backend
    tik = time.time()
    threads = []
    results = [None] * args.n_thread
    for i in range(args.n_thread):
        t = threading.Thread(target=send_request, args=(results, i))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"Time (POST): {time.time() - tik} s")
    # print(prompt.replace(conv.sep, "\n"), end="")
    n_words = 0
    # for i, result in enumerate(results):
    #     # print(result.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"))
    #     #print(f"It takes {time.time() - tik}")
    #     #print(json.loads(k[-2].decode("utf-8"))["text"])
    #     # json.loads(k[-2].decode("utf-8"))["text"]
    #     n_words += len(result.json()["text"]) - prompt_len

    # if streaming:
    for i, response in enumerate(results):
        # print(prompt[i].replace(conv.sep, "\n"), end="")
        k = list(response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"))
        response_new_words = json.loads(k[-2].decode("utf-8"))["text"]
        n_words += len(response_new_words.split(" ")) - len(prompt[i].split(" "))
        # for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
        #     if chunk:
        #         data = json.loads(chunk.decode("utf-8"))
        #         output = data["text"].split(conv.sep)[-1]
        #         # print(output, end="\r")
        #         print(output)

    time_seconds = time.time() - tik
    print(f"Time (total): {time_seconds} to finish, n threads: {args.n_thread}, "
          f"throughput: {n_words / time_seconds} words/s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-address", type=str, default="http://localhost:21001")
    parser.add_argument("--worker-address", type=str)
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--message", type=str, default=
        "Tell me a story with more than 2000 words.")
    parser.add_argument("--n-thread", type=int, default=2)
    args = parser.parse_args()

    main()
