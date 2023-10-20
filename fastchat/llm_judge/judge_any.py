import json
import shortuuid
import yaml
import argparse
import os
import openai
import concurrent.futures
from tqdm import tqdm
from string import Formatter

from fastchat.llm_judge.common import (
    load_questions,
    chat_compeletion_openai,
    chat_compeletion_anthropic,
    chat_compeletion_palm,
    load_questions,
    load_model_answers,
)

from fastchat.model.model_adapter import get_conversation_template


# get answer from model
def get_answer(model, conv, temperature, max_tokens, api):
    if model in ["claude-v1", "claude-instant-v1", "claude-2"]:
        output = chat_compeletion_anthropic(
            model, conv, temperature, max_tokens
        )
    elif model == "palm-2-chat-bison-001":
        chat_state, output = chat_compeletion_palm(
            chat_state, model, conv, temperature, max_tokens
        )
    else:
        output = chat_compeletion_openai(model, conv, temperature, max_tokens)
    return output


# load config args from config yaml files
def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    return config_kwargs


# question: dict, answer: dict, reference: dict, configs: dict, output_file: str
def judgment(**args):
    question = args["question"]
    answer = args["answer"]
    reference = args["reference"]
    configs = args["configs"]
    output_file = args["output_file"]

    model = configs["judge_model"]
    prompt_args = {}
    
    for i, turn in enumerate(question["turns"]):
        prompt_args[f"question_{i+1}"] = turn

    if answer:
        for i, turn in enumerate(answer["choices"][0]["turns"]):
            prompt_args[f"answer_{i+1}"] = turn
    if reference:
        for i, turn in enumerate(reference["choices"][0]["turns"]):
            prompt_args[f"ref_answer_{i+1}"] = turn
    
    user_prompt = configs["prompt_template"].format(**prompt_args)

    conv = get_conversation_template(model)
    conv.set_system_message(configs["system_prompt"])

    # add few shot prompts
    if configs["few_shot"]:
        for i, shot in enumerate(configs["few_shot"]):
            conv.append_message(conv.roles[i % 2], shot)

    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)

    judgment = get_answer(model, conv, configs["temperature"], configs["max_tokens"], configs["api_base_list"])

    with open(output_file, "a") as f:
        output = {
            "question_id": question["question_id"],
            # "answer_id": shortuuid.uuid(),
            "model_id": answer["model_id"],
            "prompt": conv.messages,
            "judgment": [{"index": 0, "turns":[judgment]}],
        }

        f.write(json.dumps(output, ensure_ascii=False) + "\n")


def turn_by_turn_judgment(**args):
    configs = args["configs"]
    reference = args["reference"]

    model = configs["judge_model"]
    judgments = []
    convs = []

    for i, turn in enumerate(args["iterable"]):
        conv = get_conversation_template(model)
        conv.set_system_message(configs["system_prompt"])

        prompt_args = {}
        if reference:
            assert reference[i]
            prompt_args[f"ref_answer_{1}"] = reference[i]

        prompt_args[f"question_{1}"] = turn
        user_prompt = configs["prompt_template"].format(**prompt_args)

        # add few shot prompts
        if configs["few_shot"]:
            for i, shot in enumerate(configs["few_shot"]):
                conv.append_message(conv.roles[i % 2], shot)

        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)
        
        output = get_answer(model, conv, configs["temperature"], configs["max_tokens"], configs["api_base_list"])
        
        judgments.append(output)
        convs.append(conv.messages)
    
    with open(args["output_file"], "a") as f:
        output = {}
        output["question_id"] = args["question_id"]
        if args["answer_id"]:
            output["answer_id"] = args["answer_id"]
        else:
            output["answer_id"] = shortuuid.uuid()
        output["model_id"] = args["model_id"]
        output["judge_answer"] = judgments
        output["prompt"] = convs

        f.write(json.dumps(output, ensure_ascii=False) + "\n")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--bench-name", type=str, required=True)
    parser.add_argument(
        "--mode", 
        type=str,
        default="all",
        choices=["all", "question", "answer"],
        help=(
            "Judge mode."
            "`all` judge the entire conversation."
            "`question` judge prompts only turn-by-turn"
            "`answer` judge model answers only turn-by-turn"
        ),
        required=True
    )
    parser.add_argument("--model", type=str, default=None, help="Specify a model's response to be judged.")
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument("--output-file", type=str, default=None)
    args = parser.parse_args()

    configs = make_config(args.config_file)

    if configs["api_base_list"]:
        openai.api_base = configs["api_base_list"]

    question_file = os.path.join("data", args.bench_name, "question.jsonl")
    answer_dir = os.path.join("data", args.bench_name, "model_answer")
    ref_answer_dir = os.path.join("data", args.bench_name, "reference_answer")

    questions = load_questions(question_file, None, None)
    model_answers = load_model_answers(answer_dir)

    if args.model:
        model_answers = model_answers[args.model]

    # if mode isn't all, then need to create a iterable list for turn-by-turn judgment
    if args.mode == "question":
        iterables = []
        for prompt in questions:
            obj = {}
            obj["iterables"] = prompt["turns"]
            if "question_id" in prompt:
                obj["question_id"] = prompt["question_id"]
            else:
                obj["question_id"] = None
            if "answer_id" in prompt:
                obj["answer_id"] = prompt["answer_id"]
            else:
                obj["answer_id"] = None
            if "model_id" in prompt:
                obj["model_id"] = prompt["model_id"]
            else:
                obj["model_id"] = configs["judge_model"]
            iterables.append(obj) 
    elif args.mode == "answer":
        iterables = []
        for model in model_answers.keys():
            for i in range(1, len(model_answers[model])+1):
                prompt = model_answers[i]
                obj = {}
                obj["iterables"] = prompt["choices"][0]["turns"]
                if "question_id" in prompt:
                    obj["question_id"] = prompt["question_id"]
                else:
                    obj["question_id"] = None
                if "answer_id" in prompt:
                    obj["answer_id"] = prompt["answer_id"]
                else:
                    obj["answer_id"] = None
                if "model_id" in prompt:
                    obj["model_id"] = prompt["model_id"]
                else:
                    obj["model_id"] = None
                iterables.append(obj)  

    ref_answers = None
    if configs["reference"]:
        ref_answers = load_model_answers(ref_answer_dir)
        ref_answers = ref_answers[configs["ref_model"]]

    if args.output_file:
        output_file = args.output_file
    else:
        output_file = os.path.join(
                "data", 
                args.bench_name,
                "model_judgment",
                f"{configs['name']}_judge_any.jsonl"
            )
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        if args.mode == "all":
            for model in model_answers.keys():
                for question in questions:
                    kwargs = {}
                    kwargs["question"] = question
                    kwargs["answer"] = model_answers[model][question["question_id"]]
                    if ref_answers:
                        kwargs["reference"] = ref_answers[question["question_id"]]
                    else:
                        kwargs["reference"] = None

                    kwargs["configs"] = configs
                    kwargs["output_file"] = output_file
                    future = executor.submit(
                        judgment,
                        **kwargs
                    )
                    futures.append(future)
            for future in tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                future.result()
        else:
            for iter in iterables:
                kwargs = {}
                kwargs["iterable"] = iter["iterables"]      
                kwargs["question_id"] = iter["question_id"]
                kwargs["answer_id"] = iter["answer_id"]
                kwargs["model_id"] = iter["model_id"]

                if ref_answers:
                    kwargs["reference"] = ref_answers[iter["question_id"]]
                else:
                    kwargs["reference"] = None

                kwargs["configs"] = configs
                kwargs["output_file"] = output_file
                future = executor.submit(
                    turn_by_turn_judgment,
                    **kwargs
                )
                futures.append(future)
            for future in tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                future.result()