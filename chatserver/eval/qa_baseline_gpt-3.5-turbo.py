"""Generate answers with GPT-3.5"""
# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import argparse
import json
import os
import time

import openai
import tqdm


def get_eval(question: str, max_tokens: int):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{
            'role': 'system',
            'content': 'You are a helpful assistant.'
        }, {
            'role': 'user',
            'content': question,
        }],
        max_tokens=max_tokens,
    )
    return response['choices'][0]['message']['content']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT answer generation.')
    parser.add_argument('-q', '--question')
    parser.add_argument('-o', '--output')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    with open(os.path.expanduser(args.question)) as f:
        question = json.load(f)
        questions_dict = {q['id']: q['question'] for q in question['questions']}

    answers = []

    for qid, question in tqdm.tqdm(questions_dict.items()):
        for retries in range(3):
            try:
                eval_result = get_eval(question, args.max_tokens)
                answers.append({'id': qid, 'answer': eval_result})
                break
            except Exception as e:
                print('Error: ', e)
                if retries == 2:
                    answers.append({'id': qid, 'answer': '#ERROR#'})

    with open(os.path.expanduser(args.output), 'w') as f:
        json.dump({'model': 'gpt-3.5-turbo', 'answers': answers}, f)
