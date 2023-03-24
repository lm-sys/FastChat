"""Evaluate QA with ChatGPT."""
# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import argparse
import json
import os
import time

import openai
import tqdm

import ray

@ray.remote
def get_eval(rule: str, user: str, assistant1: str, assistant2: str, max_tokens: int):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{
            'role': 'system',
            'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
        }, {
            'role': 'user',
            'content': (f'[User Question]\n{user}\n\n[Assistant 1]\n{assistant1}\n\n'
                        f'[Assistant 2]\n{assistant2}\n\n[system]\n{rule}'),
        }],
        temperature=0.2,  # TODO: figure out which temperature is best for evaluation
        max_tokens=max_tokens,
    )
    return response['choices'][0]['message']['content']

@ray.remote
def get_eval_ver2(rule: str, user: str, assistant1: str, assistant2: str, max_tokens: int):
    response = openai.Completion.create(
        model='gpt-3.5-turbo',
        # messages=[{
        #     'role': 'system',
        #     'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
        # }, {
        #     'role': 'user',
        #     'content': (f'[User]\n{user}\n[Assistant 1]\n{assistant1}\n'
        #                 f'[Assistant 2]\n{assistant2}\n[system]\n{rule}'),
        # }],
        prompt=(f'[User Question]\n{user}\n[Assistant 1]\n{assistant1}\n'
                f'[Assistant 2]\n{assistant2}\n[system]\n{rule}'),
        temperature=0.2,  # TODO: figure out which temperature is best for evaluation
        max_tokens=max_tokens,
    )
    return response['choices'][0]['message']['content']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-q', '--question')
    # parser.add_argument('-a', '--answer')
    parser.add_argument('-a', '--answer-list', nargs='+', default=[])
    parser.add_argument('-r', '--rule')
    parser.add_argument('-o', '--output')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    # ray.init()

    f_q = open(os.path.expanduser(args.question))
    f_ans1 = open(os.path.expanduser(args.answer_list[0]))
    f_ans2 = open(os.path.expanduser(args.answer_list[1]))

    with open(os.path.expanduser(args.rule)) as f:
        rule = f.read()

    # prompt_file = open(f'prompt_list_alpaca_vicuna.txt', 'w')
    # prompt_file = open(f'prompt_list_gpt3.5_vicuna.txt', 'w')
    prompt_file = open(f'prompt_list_llama_vicuna.txt', 'w')

    evaluations = []
    eval_result_handle = []
    scores = [0, 0]
    idx = 1
    for ques, ans1, ans2 in zip(f_q, f_ans1, f_ans2):
        ques = json.loads(ques)["question"]
        ans1 = json.loads(ans1)["answer"]
        ans2 = json.loads(ans2)["answer"]
        
        prompt=(f'[User Question]\n{ques}\n\n[Assistant 1]\n{ans1}\n\n'
                f'[Assistant 2]\n{ans2}\n\n[System]\n{rule}\n\n'
                '=====================\n\n')
        # print(prompt)
        prompt_file.write('Q'+str(idx) + '\n\n')
        prompt_file.write(prompt)
        idx += 1
        # eval_result_handle.append(get_eval.remote(rule, ques, ans2, ans2, args.max_tokens))

    # for qid, eval_result in enumerate(ray.get(eval_result_handle)):
    #     score, explanation = eval_result.split('\n', 1)
    #     print(eval_result)
    #     evaluations.append({'id': qid, 'score': int(score), 'explanation': explanation})
    #     scores[int(score)-1] += 1
    # print(scores)
    # with open(os.path.expanduser(args.output), 'w') as f:
    #     json.dump(evaluations, f)
