"""Generate answers with local models.

Usage:
python3 gen_vllm_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import time

import shortuuid
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import get_conversation_template


def group_question_by_temperature(questions):
    """ return temperature as key, questions list as value """
    temperature2qs = {}
    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7
        if temperature not in temperature2qs:
            temperature2qs[temperature] = []
        temperature2qs[temperature].append(question)

    return temperature2qs


def get_max_num_turns(questions):
    return max(
        [len(question["turns"]) for question in questions]
    )


def gather_id_inputs(
    cur_turn_id,
    id2outputs,
    model_id,
    questions,
    num_choices,
):
    id2inputs = {}
    for question in questions:
        turns = question["turns"]
        if len(turns) < cur_turn_id:
            continue
        for i in range(num_choices):
            key = (question["question_id"], i)
            assistant_contents = id2outputs.get(key, [])
            conv = get_conversation_template(model_id)

            for j in range(len(turns[:cur_turn_id])):
                qs = turns[j]
                assistant_content = None
                if len(assistant_contents) > j:
                    assistant_content = assistant_contents[j]

                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], assistant_content)

            prompt = conv.get_prompt()
            if key not in id2inputs:
                id2inputs[key] = []
            id2inputs[key].append(prompt)

    return id2inputs


def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    gpu_memory_utilization,
    presence_penalty,
    frequency_penalty,
):
    questions = load_questions(question_file, question_begin, question_end)
    get_vllm_model_answers(
        model_path,
        model_id,
        questions,
        answer_file,
        max_new_token,
        num_choices,
        gpu_memory_utilization,
        presence_penalty,
        frequency_penalty,
    )


def get_vllm_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    gpu_memory_utilization,
    presence_penalty,
    frequency_penalty,
):
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )

    max_num_turns = get_max_num_turns(questions)
    temperature2qs = group_question_by_temperature(questions)
    id2outputs = {}
    for cur_turn_id in range(1, max_num_turns + 1):
        print(f"Process turn {cur_turn_id}")
        for temperature, sub_questions in temperature2qs.items():
            conv = get_conversation_template(model_id)
            stop_token_ids = [tokenizer.eos_token_id, tokenizer.pad_token_id]
            if conv.stop_token_ids:
                stop_token_ids.extend(conv.stop_token_ids)

            inference_params = {
                "temperature": temperature,
                "max_tokens": max_new_token,
                "stop_token_ids": stop_token_ids,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
            }
            print(f"Process {len(sub_questions)} questions with temperature {temperature}")
            sampling_params = SamplingParams(**inference_params)
            id2inputs = gather_id_inputs(
                cur_turn_id,
                id2outputs,
                model_id,
                sub_questions,
                num_choices,
            )

            batched_index = []
            batched_inputs = []
            for (quid, choice_index), inputs in id2inputs.items():
                for each_input in inputs:
                    batched_inputs.append(each_input)
                    batched_index.append((quid, choice_index))

            batched_outputs = llm.generate(batched_inputs, sampling_params)
            id2outputs = gather_outputs(id2outputs, batched_index, batched_outputs)

    quid2choices = gather_choices(id2outputs)
    write_answers(quid2choices, questions, model_id, answer_file)


def gather_outputs(id2outputs, batched_index, batched_outputs):
    for idx, output in enumerate(batched_outputs):
        generated_text = output.outputs[0].text
        (quid, choice_index) = batched_index[idx]
        if (quid, choice_index) not in id2outputs:
            id2outputs[(quid, choice_index)] = []
        id2outputs[(quid, choice_index)].append(generated_text)
    return id2outputs


def gather_choices(id2outputs):
    quid2choices = {}
    for (quid, choice_index) in id2outputs:
        turns = id2outputs[(quid, choice_index)]
        if quid not in quid2choices:
            quid2choices[quid] = []
        quid2choices[quid].append({"index": choice_index, "turns": turns})
    return quid2choices


def write_answers(quid2choices, questions, model_id, answer_file):
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    for question in questions:
        question_id = question["question_id"]
        choices = quid2choices[question_id]
        with open(os.path.expanduser(answer_file), "a", encoding="utf-8") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "category": question["category"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json, ensure_ascii=False) + "\n")



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
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="The ratio (between 0 and 1) of GPU memory to"
             "reserve for the model weights, activations, and KV cache.",
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0,
        help="""Float that penalizes new tokens based on whether they
                appear in the generated text so far. Values > 0 encourage the llm
                to use new tokens, while values < 0 encourage the llm to repeat
                tokens.""",
    )
    parser.add_argument(
        "--frequency_penalty",
        type=float,
        default=0,
        help="""Float that penalizes new tokens based on their
                frequency in the generated text so far. Values > 0 encourage the
                llm to use new tokens, while values < 0 encourage the llm to
                repeat tokens.""",
    )

    args = parser.parse_args()

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        model_path=args.model_path,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        gpu_memory_utilization=args.gpu_memory_utilization,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
    )

    reorg_answer_file(answer_file)
