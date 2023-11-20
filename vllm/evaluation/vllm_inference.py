"""Generate answers with local models.

Usage:
python3 gen_model_answer_01.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import random
import string
import argparse
import json
import os
import random
import re
import time
from tensor_parallel import TensorParallelPreTrainedModel
import shortuuid
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from common import load_questions, temperature_config, get_all_filenames
from model import load_model, get_conversation_template
from common import should_process_file
from vllm import LLM, SamplingParams
from tensor_parallel import TensorParallelPreTrainedModel


def generate_random_string(length=23):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


def process_file(model_id, questions, path,identifier, max_new_token, num_choices, data_id, model, tokenizer, max_token):
    answer_file=path+identifier+".jsonl"
    prompts = []
    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template(model_id)

            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                prompts.append(prompt)
    temperature = 0.4
    sampling_params = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_token)
    output_ids = model.generate(
        prompts,
        sampling_params,
    )
    
    for output_id, output in enumerate(output_ids):
        output = output.outputs[0].text

        if conv.stop_str and output.find(conv.stop_str) > 0:
            output = output[: output.find(conv.stop_str)]

        if conv.name == "xgen" and output.startswith("Assistant:"):
            output = output.replace("Assistant:", "", 1).strip()

        with open(answer_file, "a", encoding='utf-8') as fout:

            ans_json = {
                "question_id": int(questions[output_id]["question_id"]),
                "answer_id": generate_random_string(),
                "model_id": model_id,
                "choices": [{"index": 0, "turns": [output]}],
                "instruction": questions[output_id]["turns"][0],
            }
            json.dump(ans_json, fout, ensure_ascii=False)
            fout.write("\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="The path to the data",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="The path to save result",
    )
    parser.add_argument(
        "--card-id",
        type=int,
        default=0,
        help="The path to save result",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="The number used in inference",
    )
    parser.add_argument(
        "--identifier",
        type=str,
        default="",
        help="The name to save result",
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
        "--max_token",
        type=int,
        default=1024,
        help="output length",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=8, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    args = parser.parse_args()
    model = LLM(model=args.model_path, trust_remote_code=True, tensor_parallel_size=args.tensor_parallel_size)

    filenames = get_all_filenames(args.data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # import pdb; pdb.set_trace()
    for i, filename in enumerate(filenames):
        print(
            f"------------------------------------now is the {i}/{len(filenames)}------------------------------------")
        # print(should_process_file(args.card_id,int(filename[:-6])))
        if os.path.exists(args.output_path+args.identifier+".jsonl"):
            continue
        questions = load_questions(args.data_path + filename, args.question_begin, args.question_end, tokenizer,
                                   model_id=args.model_id, )
        process_file(args.model_id, questions, args.output_path,args.identifier, args.max_new_token, args.num_choices, filename, model,
                     tokenizer, args.max_token)
