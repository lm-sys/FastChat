import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
from tqdm import tqdm
import ray

from chatserver.conversation import default_conversation
from chatserver.utils import disable_torch_init

@ray.remote(num_gpus=1)
@torch.inference_mode()
def eval_model(model_name, questions_file, answers_file):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
        torch_dtype=torch.float16).cuda()


    qa_file = open(os.path.expanduser(questions_file), "r")
    ans_file = open(os.path.expanduser(answers_file), "w")
    for i, line in enumerate(tqdm(qa_file)):
        idx = json.loads(line)["id"]
        qs = json.loads(line)["question"]
        cat = json.loads(line)["category"]
        conv = default_conversation.copy()
        conv.append_message(conv.roles[0], qs)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])
        output_ids = model.generate(
            torch.as_tensor(inputs.input_ids).cuda(),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=1024)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        try:
            index = outputs.index(conv.sep, len(prompt))
        except ValueError:
            outputs += conv.sep
            index = outputs.index(conv.sep, len(prompt))

        outputs = outputs[len(prompt) + len(conv.roles[1]) + 2:index].strip()
        ans_file.write(json.dumps({"id": idx, "answer": outputs, "category": cat}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--questions-file", type=str, default="mini_evals/qa.jsonl")
    parser.add_argument("--answers-file", type=str, default="answers.jsonl")
    args = parser.parse_args()

    ray.init()
    handle = []
    for i in range(1, 5):
        model_name = args.model_name
        model_name.replace('~/', '')
        print(model_name)
        question_file = f'mini_evals/qa_v2-{i}.jsonl'
        answers_file = f'answers/v4/answers-v2-{i}.jsonl'
        handle.append(eval_model.remote(model_name, question_file, answers_file))

    results = ray.get(handle)
