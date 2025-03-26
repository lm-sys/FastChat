"""
A model worker that executes the model based on IPEX-LLM.

See documentations at docs/ipex_llm_integration.md
"""


import torch
import torch.nn.functional as F
import gc
import argparse
import asyncio
import atexit
import json
from typing import List
import uuid
from threading import Thread
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.serve.model_worker import (
    logger,
    worker_id,
)
from fastchat.serve.base_model_worker import (
    create_background_tasks,
    acquire_worker_semaphore,
    release_worker_semaphore,
)
from fastchat.utils import get_context_length, is_partial_stop

from ipex_llm.transformers.loader import load_model
from transformers import TextIteratorStreamer

app = FastAPI()


class BigDLLLMWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        conv_template: str = None,
        load_in_low_bit: str = "sym_int4",
        device: str = "cpu",
        no_register: bool = False,
        trust_remote_code: bool = False,
        embed_in_truncate: bool = False,
        speculative: bool = False,
        load_low_bit_model: bool = False,
        stream_interval: int = 4,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template,
        )

        self.load_in_low_bit = load_in_low_bit
        self.load_low_bit_model = load_low_bit_model
        logger.info(
            f"Loading the model {self.model_names} on worker {worker_id},"
            f" worker type: BigDLLLM worker..."
        )

        logger.info(f"Using low bit format: {self.load_in_low_bit}, device: {device}")
        if speculative:
            logger.info(f"Using Self-Speculative decoding to generate")

        self.device = device
        self.speculative = speculative
        self.model, self.tokenizer = load_model(
            model_path,
            device,
            self.load_in_low_bit,
            trust_remote_code,
            speculative,
            load_low_bit_model,
        )
        self.stream_interval = stream_interval
        self.context_len = get_context_length(self.model.config)
        self.embed_in_truncate = embed_in_truncate
        if not no_register:
            self.init_heart_beat()

    def __process_embed_chunk(self, input_ids, attention_mask, **model_type_dict):
        if model_type_dict.get("is_bert"):
            model_output = self.model(input_ids)
            if model_type_dict.get("is_robert"):
                data = model_output.last_hidden_state
            else:
                data = model_output[0]
        elif model_type_dict.get("is_t5"):
            model_output = self.model(input_ids, decoder_input_ids=input_ids)
            data = model_output.encoder_last_hidden_state
        else:
            model_output = self.model(input_ids, output_hidden_states=True)
            if model_type_dict.get("is_chatglm"):
                data = model_output.hidden_states[-1].transpose(0, 1)
            else:
                data = model_output.hidden_states[-1]

        if hasattr(self.model, "use_cls_pooling") and self.model.use_cls_pooling:
            sum_embeddings = data[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).expand(data.size()).float()
            masked_embeddings = data * mask
            sum_embeddings = torch.sum(masked_embeddings, dim=1)
        token_num = torch.sum(attention_mask).item()

        return sum_embeddings, token_num

    @torch.inference_mode()
    def get_embeddings(self, params):
        self.call_ct += 1

        try:
            # Get tokenizer
            tokenizer = self.tokenizer
            ret = {"embedding": [], "token_num": 0}

            # Based on conditions of different model_type
            model_type_dict = {
                "is_llama": "llama" in str(type(self.model)),
                "is_t5": "t5" in str(type(self.model)),
                "is_chatglm": "chatglm" in str(type(self.model)),
                "is_bert": "bert" in str(type(self.model)),
                "is_robert": "robert" in str(type(self.model)),
            }

            if self.embed_in_truncate:
                encoding = tokenizer.batch_encode_plus(
                    params["input"],
                    padding=True,
                    truncation="longest_first",
                    return_tensors="pt",
                    max_length=self.context_len,
                )
            else:
                encoding = tokenizer.batch_encode_plus(
                    params["input"], padding=True, return_tensors="pt"
                )
            input_ids = encoding["input_ids"].to(self.device)
            # Check if we need attention_mask or not.
            attention_mask = input_ids != tokenizer.pad_token_id

            base64_encode = params.get("encoding_format", None)

            if self.embed_in_truncate:
                embedding, token_num = self.__process_embed_chunk(
                    input_ids, attention_mask, **model_type_dict
                )
                if (
                    not hasattr(self.model, "use_cls_pooling")
                    or not self.model.use_cls_pooling
                ):
                    embedding = embedding / token_num
                normalized_embeddings = F.normalize(embedding, p=2, dim=1)
                ret["token_num"] = token_num
            else:
                all_embeddings = []
                all_token_num = 0
                for i in range(0, input_ids.size(1), self.context_len):
                    chunk_input_ids = input_ids[:, i:i + self.context_len]
                    chunk_attention_mask = attention_mask[:, i:i + self.context_len]

                    # add cls token and mask to get cls embedding
                    if (
                        hasattr(self.model, "use_cls_pooling")
                        and self.model.use_cls_pooling
                    ):
                        cls_tokens = (
                            torch.zeros(
                                (chunk_input_ids.size(0), 1),
                                dtype=chunk_input_ids.dtype,
                                device=chunk_input_ids.device,
                            )
                            + tokenizer.cls_token_id
                        )
                        chunk_input_ids = torch.cat(
                            [cls_tokens, chunk_input_ids], dim=-1
                        )
                        mask = torch.ones(
                            (chunk_attention_mask.size(0), 1),
                            dtype=chunk_attention_mask.dtype,
                            device=chunk_attention_mask.device,
                        )
                        chunk_attention_mask = torch.cat(
                            [mask, chunk_attention_mask], dim=-1
                        )

                    chunk_embeddings, token_num = self.__process_embed_chunk(
                        chunk_input_ids, chunk_attention_mask, **model_type_dict
                    )
                    if (
                        hasattr(self.model, "use_cls_pooling")
                        and self.model.use_cls_pooling
                    ):
                        all_embeddings.append(chunk_embeddings * token_num)
                    else:
                        all_embeddings.append(chunk_embeddings)
                    all_token_num += token_num

                all_embeddings_tensor = torch.stack(all_embeddings)
                embedding = torch.sum(all_embeddings_tensor, dim=0) / all_token_num
                normalized_embeddings = F.normalize(embedding, p=2, dim=1)

                ret["token_num"] = all_token_num

            if base64_encode == "base64":
                out_embeddings = self.__encode_base64(normalized_embeddings)
            else:
                out_embeddings = normalized_embeddings.tolist()
            ret["embedding"] = out_embeddings

            gc.collect()
            torch.cuda.empty_cache()
            if self.device == "xpu":
                torch.xpu.empty_cache()
            if self.device == "npu":
                torch.npu.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
        return ret

    def generate_stream_gate(self, params):
        self.call_ct += 1
        # context length is self.context_length
        prompt = params["prompt"]
        temperature = float(params.get("temperature", 1.0))
        do_sample = bool(params.get("do_sample", False))
        repetition_penalty = float(params.get("repetition_penalty", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = int(params.get("top_k", 1))
        if top_k == -1:
            top_k = 1
        max_new_tokens = int(params.get("max_new_tokens", 256))
        echo = bool(params.get("echo", True))
        stop_str = params.get("stop", None)
        stop_token_ids = params.get("stop_token_ids", None) or []
        if self.tokenizer.eos_token_id not in stop_token_ids:
            stop_token_ids.append(self.tokenizer.eos_token_id)

        # Handle stop_str
        stop = set()
        if isinstance(stop_str, str) and stop_str != "":
            stop.add(stop_str)
        elif isinstance(stop_str, list) and stop_str != []:
            stop.update(stop_str)

        for tid in stop_token_ids:
            if tid is not None:
                s = self.tokenizer.decode(tid)
                if s != "":
                    stop.add(s)

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        if self.device == "xpu":
            input_ids = input_ids.to("xpu")

        input_echo_len = input_ids.shape[1]

        if self.model.config.is_encoder_decoder:
            max_src_len = self.context_len
            input_ids = input_ids[:max_src_len]
            input_echo_len = len(input_ids)
            prompt = self.tokenizer.decode(
                input_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
        else:
            # Truncate the max_new_tokens if input_ids is too long
            new_max_new_tokens = min(self.context_len - input_echo_len, max_new_tokens)
            if new_max_new_tokens < max_new_tokens:
                logger.info(
                    f"Warning: max_new_tokens[{max_new_tokens}] + prompt[{input_echo_len}] greater "
                    f"than context_length[{self.context_len}]"
                )
                logger.info(f"Reset max_new_tokens to {new_max_new_tokens}")
                max_new_tokens = new_max_new_tokens

        # Use TextIteratorStreamer for streaming output
        streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer,
            timeout=60,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Generation config:
        # https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig
        generated_kwargs = dict(
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
        )

        def model_generate():
            self.model.generate(input_ids, **generated_kwargs)

        t1 = Thread(target=model_generate)
        t1.start()

        stopped = False
        finish_reason = None
        if echo:
            partial_output = prompt
            rfind_start = len(prompt)
        else:
            partial_output = ""
            rfind_start = 0

        for i in range(max_new_tokens):
            try:
                output_token = next(streamer)
            except StopIteration:
                # Stop early
                stopped = True
                break
            partial_output += output_token

            if i % self.stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                for each_stop in stop:
                    pos = partial_output.rfind(each_stop, rfind_start)
                if pos != -1:
                    partial_output = partial_output[:pos]
                    stopped = True
                    break
                else:
                    partially_stopped = is_partial_stop(partial_output, each_stop)
                    if partially_stopped:
                        break
                if not partially_stopped:
                    json_output = {
                        "text": partial_output,
                        "usage": {
                            "prompt_tokens": input_echo_len,
                            "completion_tokens": i,
                            "total_tokens": input_echo_len + i,
                        },
                        "finish_reason": None,
                    }
                    ret = {
                        "text": json_output["text"],
                        "error_code": 0,
                    }
                    ret["usage"] = json_output["usage"]
                    ret["finish_reason"] = json_output["finish_reason"]
                    yield json.dumps(ret).encode() + b"\0"

            if stopped:
                break
        else:
            finish_reason = "length"

        if stopped:
            finish_reason = "stop"
        json_output = {
            "text": partial_output,
            "error_code": 0,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": i,
                "total_tokens": input_echo_len + i,
            },
            "finish_reason": finish_reason,
        }
        yield json.dumps(json_output).encode() + b"\0"

    def generate_gate(self, params):
        for x in self.generate_stream_gate(params):
            # for x in self.generate_stream2(params):
            pass
        return json.loads(x[:-1].decode())


# Below are api interfaces
@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    generator = worker.generate_stream_gate(params)
    background_tasks = create_background_tasks()
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    output = await asyncio.to_thread(worker.generate_gate, params)
    release_worker_semaphore()
    return JSONResponse(output)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.post("/worker_get_embeddings")
async def api_get_embeddings(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    embedding = worker.get_embeddings(params)
    release_worker_semaphore()
    return JSONResponse(content=embedding)


@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    return worker.count_token(params)


@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    return worker.get_conv_template()


@app.post("/model_details")
async def api_model_details(request: Request):
    return {"context_length": worker.context_len}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--model-path", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--limit-worker-concurrency", type=int, default=1024)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument("--stream-interval", type=int, default=4)
    parser.add_argument(
        "--low-bit", type=str, default="sym_int4", help="Low bit format."
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for executing model, cpu/xpu"
    )
    parser.add_argument(
        "--speculative",
        action="store_true",
        default=False,
        help="To use self-speculative or not",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Trust remote code (e.g., from HuggingFace) when"
        "downloading the model and tokenizer.",
    )
    parser.add_argument(
        "--load-low-bit-model",
        action="store_true",
        default=False,
        help="Load models that have been converted/saved using ipex-llm's save_low_bit interface",
    )
    parser.add_argument("--embed-in-truncate", action="store_true")

    args = parser.parse_args()
    worker = BigDLLLMWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        args.conv_template,
        args.low_bit,
        args.device,
        args.no_register,
        args.trust_remote_code,
        args.embed_in_truncate,
        args.speculative,
        args.load_low_bit_model,
        args.stream_interval,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
