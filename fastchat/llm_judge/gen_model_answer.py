"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time
import requests
import re
from load import Xgen15BTokenizer
import transformers
from transformers import AutoModelForCausalLM

import shortuuid
import torch
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype


def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
    revision,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                dtype=dtype,
                revision=revision,
            )
        )

    if use_ray:
        ray.get(ans_handles)

def fetch_model(model_pth):
    model = AutoModelForCausalLM.from_pretrained(
        model_pth,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
        device_map='auto',
    )
    tokenizer = Xgen15BTokenizer()
    return model, tokenizer

def parse_response(response, model_id):
    # split based on STOPS
    if model_id.lower() == "Mixtral-8x7b".lower():
        STOPS = ["<s>", "</s>", "[INST]", "[/INST]"]
    else:
        STOPS = ["<|system|>", "<|user|>", "<|assistant|>", "<|endofprompt|>", "<|endoftext|>"]

    # Join the stops using "|" for regex OR operator
    stops_pattern = "|".join(map(re.escape, STOPS))

    # Split the string using stops_pattern
    result = re.split(stops_pattern, response)
    return result[-1]

def generate_endpoint(prompt, model):
    if model.lower() == "XGen-22b-v2".lower():
        url = "https://inference.salesforceresearch.ai/generate"
    elif model.lower() == "Mixtral-8x7b".lower():
        url = "https://mixtralai.salesforceresearch.ai/generate"
    else:
        raise ValueError(f"Model {model} not supported")
    print(prompt)
    data = {
        "prompt": prompt,
        "use_beam_search": False,
        "n": 1,
        "temperature": 0.0,
        "max_tokens": 2000,
        "stop" : ['<|endofprompt|>', '[/RESPONSE]']
    }
    response = requests.post(url, json=data)

    return parse_response(response.json()['text'][0], model)

def format_prompt(messages, model):
    formatted_prompt = ""

    # Tony's finetuned models
    if model.endswith("pm"):
        for message in messages:
            if message["role"] == "user":
                formatted_prompt += f"<|im_start|>user\n{message['content']}<|im_end|>\n"
            elif message["role"] == "assistant":
                formatted_prompt += f"<|im_start|>assistant\n{message['content']}<|im_end|>\n"

        formatted_prompt += '<|im_start|>assistant\n'

    # Mixtral endpoint
    elif model.lower() == "Mixtral-8x7b".lower():
        for message in messages:
            if message["role"] == "user":
                formatted_prompt += f"<s>\n[INST] {message['content']} [/INST]\n"
            elif message["role"] == "assistant":
                formatted_prompt += f"{message['content']} </s>"

    # XGen template
    else:
        for message in messages:
            if message["role"] == "user":
                formatted_prompt += f"<|user|>\n{message['content']}<|endofprompt|>\n"
            elif message["role"] == "assistant":
                formatted_prompt += f"<|assistant|>\n{message['content']}<|endofprompt|>\n"

        formatted_prompt += f"<|assistant|>\n"

    return formatted_prompt


def chat(model, model_id, tokenizer, messages):
    if model_id.endswith("pm"):
        # Tony's models (OpenAI format)
        eos_token = "<|im_end|>"
    else:
        # All other models (Xgen format)
        eos_token = "<|endofprompt|>"

    input_ids = tokenizer.encode(messages, return_tensors='pt').to('cuda')

    # Generate a response
    with torch.no_grad():
        response_ids = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=2000,
            eos_token_id=tokenizer.encode(eos_token)[0]
        ).to('cuda')

    # Decode the response
    response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)[:-1]
    response = response.rstrip()
    response = parse_response(response, model_id)
    return response

@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    revision,
):
    model, tokenizer = fetch_model(model_path)

    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            messages = []
            turns = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                messages.append(
                    {"content": qs, "role": "user"}
                )
                formatted_prompt = format_prompt(messages, model_id)

                # some models may error out when generating long outputs
                try:
                    output = chat(model, model_id, tokenizer, formatted_prompt).rstrip()
                    messages.append(
                        {"content": output, "role": "assistant"}
                    )

                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR"

                turns.append(output)

            choices.append({"index": i, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )

    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    import ipdb, traceback, sys

    try:
        run_eval(
            model_path=args.model_path,
            model_id=args.model_id,
            question_file=question_file,
            question_begin=args.question_begin,
            question_end=args.question_end,
            answer_file=answer_file,
            max_new_token=args.max_new_token,
            num_choices=args.num_choices,
            num_gpus_per_model=args.num_gpus_per_model,
            num_gpus_total=args.num_gpus_total,
            max_gpu_memory=args.max_gpu_memory,
            dtype=str_to_torch_dtype(args.dtype),
            revision=args.revision,
        )

        reorg_answer_file(answer_file)
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)