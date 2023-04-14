import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fastchat.conversation import get_default_conv_template, SeparatorStyle

import requests

params = {
    "port": 5023,
}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/v1/models":
            self.send_response(200)
            self.end_headers()
            ret = requests.post(controller_addr + "/refresh_all_workers")
            ret = requests.post(controller_addr + "/list_models")
            models = ret.json()["models"]
            models.sort()
            response = json.dumps({"data": {"id": m for m in models}})

            self.wfile.write(response.encode("utf-8"))
        else:
            self.send_error(404)

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        content = self.rfile.read(content_length).decode("utf-8")
        body = json.loads(content)

        if self.path == "/api/v1/chat/completions":
            model = body["model"]
            worker_addr_res = requests.post(controller_addr + "/get_worker_address", json={"model": model})
            worker_addr = worker_addr_res.json()["address"]
            assert worker_addr != ""

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            conv = get_default_conv_template(model).copy()
            messages = body["messages"]
            context = body.get("system", conv.system)
            if context:
                # the context prompt
                conv.system = context

            conv.messages = []
            for message in messages:
                # remap roles
                username = message["role"]
                if username == "user":
                    role = conv.roles[0]
                elif username == "assistant":
                    role = conv.roles[1]
                else:
                    raise NotImplementedError("unknown role")
                sent = message["content"]
                conv.append_message(role, sent)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            max_content = body.get("max_content_length", 2048)

            n = body.get("n", 1)
            headers = {"User-Agent": "fastchat Client"}
            pload = {
                "model": model,
                "prompt": prompt,
                "max_new_tokens": max_content,
                "temperature": float(body.get("temperature", 0.5)),
                "top_p": float(body.get("top_p", 1)),
                "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
            }
            # todo make possible to parameterize this all
            # question=prompt,
            # max_new_tokens=int(body.get("max_length", 512)),
            # do_sample=bool(body.get("do_sample", True)),
            # typical_p=float(body.get("typical_p", 1)),
            # # TODO is this presence_penalty, frequency_penalty or something else?
            # repetition_penalty=float(body.get("repetition_penalty", 1.1)),
            # encoder_repetition_penalty=1,
            # top_k=int(body.get("top_k", 0)),
            # min_length=int(body.get("min_length", 0)),
            # no_repeat_ngram_size=int(body.get("no_repeat_ngram_size", 0)),
            # num_beams=int(body.get("num_beams", 1)),
            # penalty_alpha=float(body.get("penalty_alpha", 0)),
            # length_penalty=float(body.get("length_penalty", 1)),
            # early_stopping=bool(body.get("early_stopping", True)),
            # seed=int(body.get("seed", -1)),
            # stopping_strings=body.get("stop", [begin_signal]),
            initial_seed = int(body.get("seed", 0))
            choices = []
            step_size = 16
            for i in range(0, n, step_size):
                batch = min(step_size, n-i)
                pload["seed"] = i + initial_seed
                pload["n"] = batch
                response = requests.post(worker_addr + "/worker_generate_stream", headers=headers,
                                         json=pload, stream=True)
                output = ["" for _ in range(batch)]
                for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                    if chunk:
                        data = json.loads(chunk.decode())
                        if data["error_code"] == 0:
                            output[data["choice"]] = data["text"][len(prompt) + 1:].strip()
                        else:
                            self.send_error(500, f"Error, code {data['error_code']}")
                    time.sleep(0.01)

                choices.extend(output)

            response = json.dumps(
                {
                    "choices": [
                        {"index": i, "message": x} for i, x in enumerate(choices)
                    ],
                    # TODO compute
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                }
            )
            self.wfile.write(response.encode("utf-8"))
        else:
            self.send_error(404)


def run_server():
    server_addr = ("0.0.0.0" if args.listen else "127.0.0.1", params["port"])
    server = ThreadingHTTPServer(server_addr, Handler)
    if args.share:
        try:
            from flask_cloudflared import _run_cloudflared

            public_url = _run_cloudflared(params["port"], params["port"] + 1)
            print(f"Starting OpenAI-esque api at {public_url}/api")
        except ImportError:
            print("You should install flask_cloudflared manually")
    else:
        print(
            f"Starting OpenAI-esque api at http://{server_addr[0]}:{server_addr[1]}/api"
        )
    server.serve_forever()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-address", type=str, default="http://localhost:21001")
    parser.add_argument("--listen", type=str)
    parser.add_argument("--share", type=bool)
    args = parser.parse_args()
    controller_addr = args.controller_address

    run_server()
