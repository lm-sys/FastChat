import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
from tqdm import tqdm

from chatserver.conversation import default_conversation
from chatserver.utils import disable_torch_init


@torch.inference_mode()
def eval_model(model_name, questions_file):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
        torch_dtype=torch.float16).cuda()

    questions = json.load(open(f"{questions_file}", "r"))["questions"]
    answers = []
    # TODO: speedup by batching
    for i, qs in enumerate(tqdm(questions)):
        qs = qs["question"]
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
        print('======')
        print(f"{conv.roles[1]}: {outputs}")
        print('======')
        answers.append({"id": i, "answer": outputs})
    ans_all = {}
    ans_all[model_name] = answers
    return ans_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--questions-file", type=str, default="mini_evals/general/questions.json")
    parser.add_argument("--answers-file", type=str, default="answers.json")
    args = parser.parse_args()

    ans_all = eval_model(args.model_name, args.questions_file)
    json.dump(ans_all, open(args.answers_file, "w"))
