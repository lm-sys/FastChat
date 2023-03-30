"""Generate answers with GPT-3.5"""
# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import argparse
import json
import os
import time
import concurrent.futures

import openai
import tqdm
import shortuuid

MODEL = 'gpt-3.5-turbo'
MODEL_ID = 'gpt-3.5-turbo:20230327'

def get_answer(question_id: int, question: str, max_tokens: int):
    ans = {
        'answer_id': shortuuid.uuid(),
        'question_id': question_id,
        'model_id': MODEL_ID,
    }
    for _ in range(3):
        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful assistant.'
                }, {
                    'role': 'user',
                    'content': question,
                }],
                max_tokens=max_tokens,
            )
            ans['text'] = response['choices'][0]['message']['content']
            return ans
        except Exception as e:
            print('[ERROR]', e)
            ans['text'] = '#ERROR#'
            time.sleep(1)
    return ans


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT answer generation.')
    parser.add_argument('-q', '--question')
    parser.add_argument('-o', '--output')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    questions_dict = {}
    with open(os.path.expanduser(args.question)) as f:
        for line in f:
            if not line:
                continue
            q = json.loads(line)
            questions_dict[q['question_id']] = q['text']

    answers = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for qid, question in questions_dict.items():
            future = executor.submit(get_answer, qid, question, args.max_tokens)
            futures.append(future)

        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            answers.append(future.result())

    answers.sort(key=lambda x: x['question_id'])

    with open(os.path.expanduser(args.output), 'w') as f:
        table = [json.dumps(ans) for ans in answers]
        f.write('\n'.join(table))
