import argparse
import time
from typing import Union
import json

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer, OPTForCausalLM
import torch
import uvicorn


app = FastAPI()


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


async def text_streamer(args):
    context = args["prompt"]
    max_new_tokens = args.get("max_new_tokens", 1024)
    stop_str = args.get("stop", None)
    temperature = float(args.get("temperature", 1.0))

    if stop_str:
        assert len(tokenizer(stop_str).input_ids) == 1
        stop_token = tokenizer(stop_str).input_ids[0]
    else:
        stop_token = None

    input_ids = tokenizer(context).input_ids
    output_ids = list(input_ids)

    for i in range(max_new_tokens):
        if i == 0:
            out = model(
                torch.as_tensor([input_ids]).cuda(), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            attention_mask = torch.ones(1, past_key_values[0][0].shape[-2] + 1).cuda()
            out = model(input_ids=torch.as_tensor([[token]]).cuda(),
                        use_cache=True,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values)
            logits = out.logits
            past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]
        probs = torch.softmax(last_token_logits / temperature, dim=-1)
        token = int(torch.multinomial(probs, num_samples=1))
        if token == stop_token:
            break

        output_ids.append(token)
        output = tokenizer.decode(output_ids, skip_special_tokens=True)

        ret = {
            "text": output,
            "error": 0,
        }
        yield (json.dumps(ret) + "\0").encode("utf-8")


@app.post("/")
async def read_root(request: Request):
    args = await request.json()
    return StreamingResponse(text_streamer(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=10002)
    parser.add_argument("--model", type=str, default="facebook/opt-350m")
    args = parser.parse_args()

    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", add_bos_token=False)
    model = OPTForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).cuda()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
