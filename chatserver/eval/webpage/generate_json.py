"""Generate json file for webpage."""
import json
import os

models = ['llama', 'alpaca', 'gpt35', 'bard']


def read_jsonl(path: str, to_dict: bool = True):
    data = []
    with open(os.path.expanduser(path)) as f:
        for line in f:
            if not line:
                continue
            data.append(json.loads(line))
    if to_dict:
        data.sort(key=lambda x: x['id'])
        data = {item['id']: item for item in data}
    return data


def trim_hanging_lines(s: str, n: int) -> str:
    s = s.strip()
    for _ in range(n):
        s = s.split('\n', 1)[1].strip()
    return s


if __name__ == '__main__':
    # {"id": 35, "question": xxxx}
    questions = read_jsonl('../mini_evals/qa_v2.jsonl')

    # {"id": 35, "answer": xxxx}
    alpaca_answers = read_jsonl('v4/answers-alpaca-13b.jsonl')
    bard_answers = read_jsonl('v4/answers-bard.jsonl')
    gpt35_answers = read_jsonl('v4/answers-gpt-3.5-turbo.jsonl')
    llama_answers = read_jsonl('v4/answers-hf-llama-13b.jsonl')
    vicuna_answers = read_jsonl('v4/answers-vicuna-13b-20230322-split-flash.jsonl')

    # {"id": 35, "content": xxxx, "tuple": [8, 9]}
    eval_results_alpaca = read_jsonl('v4/gpt4-reviews/reviews_alpaca_vicuna.jsonl')
    eval_results_bard = read_jsonl('v4/gpt4-reviews/reviews_bard_vicuna.jsonl')
    eval_results_gpt35 = read_jsonl('v4/gpt4-reviews/reviews_gpt35_vicuna.jsonl')
    eval_results_llama = read_jsonl('v4/gpt4-reviews/reviews_llama_vicuna.jsonl')

    records = []
    for qid in questions.keys():
        r = {
            'id': qid,
            'category': questions[qid]['category'],
            'question': questions[qid]['question'],
            'answers': {
                'alpaca': alpaca_answers[qid]['answer'],
                'llama': llama_answers[qid]['answer'],
                'bard': bard_answers[qid]['answer'],
                'gpt35': gpt35_answers[qid]['answer'],
                'vicuna': vicuna_answers[qid]['answer'],
            },
            'evaluations': {
                'alpaca': eval_results_alpaca[qid]['content'],
                'llama': eval_results_llama[qid]['content'],
                'bard': eval_results_bard[qid]['content'],
                'gpt35': eval_results_gpt35[qid]['content'],
            },
            'scores': {
                'alpaca': eval_results_alpaca[qid]['tuple'],
                'llama': eval_results_llama[qid]['tuple'],
                'bard': eval_results_bard[qid]['tuple'],
                'gpt35': eval_results_gpt35[qid]['tuple'],
            },
        }

        # cleanup data
        cleaned_evals = {}
        for k, v in r['evaluations'].items():
            v = trim_hanging_lines(v, 2)
            cleaned_evals[k] = v.replace('Assistant 1', "**Assistant 1**").replace('Assistant 2', '**Assistant 2**')

        r['evaluations'] = cleaned_evals
        records.append(r)

    data = {
        'questions': records,
        'models': models,
    }

    with open('data.json', 'w') as f:
        json.dump(data, f, indent=2)
