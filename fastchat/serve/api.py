import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

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
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            messages = body["messages"]
            roles = body.get("roles", ["Human", "Assistant"])
            begin_signal = body.get("begin_signal", "### ")
            end_signal = body.get("end_signal", "\n")
            context = body.get("system", None)
            prompt_lines = []
            if context:
                # the context prompt
                prompt_lines.append(f"{context}\n\n")
            for message in messages:
                # remap roles
                username = message["role"]
                sent = message["content"]
                input = f"{begin_signal}{username}: {sent}{end_signal}"
                prompt_lines.append(input)

            max_context = body.get("max_context_length", 2048)

            while (
                    len(prompt_lines) >= 0
                    and len(encode("\n".join(prompt_lines))) > max_context
            ):
                prompt_lines.pop(0)
            prompt_lines.append(f"{begin_signal}{roles[1]} :")

            prompt = "".join(prompt_lines)

            n = body.get("n", 1)
            choices = []
            for _ in range(n):
                generator = generate_reply(
                    question=prompt,
                    max_new_tokens=int(body.get("max_length", 512)),
                    do_sample=bool(body.get("do_sample", True)),
                    temperature=float(body.get("temperature", 0.5)),
                    top_p=float(body.get("top_p", 1)),
                    typical_p=float(body.get("typical_p", 1)),
                    # TODO is this presence_penalty, frequency_penalty or something else?
                    repetition_penalty=float(body.get("repetition_penalty", 1.1)),
                    encoder_repetition_penalty=1,
                    top_k=int(body.get("top_k", 0)),
                    min_length=int(body.get("min_length", 0)),
                    no_repeat_ngram_size=int(body.get("no_repeat_ngram_size", 0)),
                    num_beams=int(body.get("num_beams", 1)),
                    penalty_alpha=float(body.get("penalty_alpha", 0)),
                    length_penalty=float(body.get("length_penalty", 1)),
                    early_stopping=bool(body.get("early_stopping", True)),
                    seed=int(body.get("seed", -1)),
                    stopping_strings=body.get("stop", [begin_signal]),
                )
                answer = ""
                # the generator streams the result, but we are not interested in a stream
                for a in generator:
                    if isinstance(a, str):
                        answer = a
                    else:
                        answer = a[0]
                choices.append(answer)

            response = json.dumps(
                {
                    "choices": [
                        {"index": i, "message": x} for i, x in enumerate(choices)
                    ],
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
