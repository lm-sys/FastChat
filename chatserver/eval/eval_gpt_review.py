import argparse
import json
import os

import openai
import tqdm
import ray
import time

@ray.remote(num_cpus=4)
def get_eval(content: str, max_tokens: int):
    try:
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=[{
                'role': 'system',
                'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
            }, {
                'role': 'user',
                'content': content,
            }],
            temperature=0.2,  # TODO: figure out which temperature is best for evaluation
            max_tokens=max_tokens,
        )
    except Exception as e:
        print(e)
        return 'error'
    return response['choices'][0]['message']['content']


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-q', '--question')
    # parser.add_argument('-a', '--answer')
    parser.add_argument('-a', '--answer-list', nargs='+', default=[])
    parser.add_argument('-r', '--rule')
    parser.add_argument('-o', '--output')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    ray.init()

    f_q = open(os.path.expanduser(args.question))
    f_ans1 = open(os.path.expanduser(args.answer_list[0]))
    f_ans2 = open(os.path.expanduser(args.answer_list[1]))
    rule_dict = json.load(open(os.path.expanduser(args.rule), 'r'))

    review_file = open(f'{args.output}', 'w')

    js_list = []
    handles = []
    idx = 0
    for ques_js, ans1_js, ans2_js in zip(f_q, f_ans1, f_ans2):
        # if idx == 10:
        #     break

        ques = json.loads(ques_js)['question']
        ans1 = json.loads(ans1_js)['answer']
        ans2 = json.loads(ans2_js)['answer']

        category = json.loads(ques_js)['category']
        if category in rule_dict:
            rule = rule_dict[category]
        else:
            rule = rule_dict['default']
        prompt = rule['prompt']
        role = rule['role']
        content = (f'[Question]\n{ques}\n\n'
                   f'[{role} 1]\n{ans1}\n\n[End of {role} 1]\n\n'
                   f'[{role} 2]\n{ans2}\n\n[End of {role} 2]\n\n'
                   f'[System]\n{prompt}\n\n')
        js_list.append({
            'id': idx+1,
            'question': ques,
            'answer1': ans1,
            'answer2': ans2,
            'category': category})
        idx += 1
        handles.append(get_eval.remote(content, args.max_tokens))
        # To avoid the rate limit set by OpenAI
        time.sleep(10)

    reviews = ray.get(handles)
    for idx, review in enumerate(reviews):
        scores = parse_score(review)
        js_list[idx]['content'] = review
        js_list[idx]['tuple'] = scores
        review_file.write(json.dumps(js_list[idx]) + '\n')
    review_file.close()
