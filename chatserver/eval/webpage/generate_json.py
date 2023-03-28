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


if __name__ == '__main__':
    # {"id": 35, "question": xxxx}
    questions = read_jsonl('../mini_evals/qa.jsonl')

    # {"id": 35, "answer": xxxx}
    gpt35_answers = read_jsonl('v3/answers-gpt-3.5-turbo.jsonl')
    alpaca_answers = read_jsonl('v3/answers-alpaca.jsonl')
    llama_answers = read_jsonl('v3/answers-hf-llama.jsonl')
    bard_answers = read_jsonl('v3/answers-bard.jsonl')
    vicuna_answers = read_jsonl('v3/answers-vicuna-v2.jsonl')

    # {"id": 35, "content": xxxx, "tuple": [8, 9]}
    eval_results_alpaca = read_jsonl('v3/result/results_alpaca.jsonl')
    eval_results_bard = read_jsonl('v3/result/results_bard.jsonl')
    eval_results_llama = read_jsonl('v3/result/results_llama.jsonl')
    eval_results_gpt35 = read_jsonl('v3/result/results_gpt35.jsonl')

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
            lines = v.split('\n')
            new_lines = []
            for line in lines:
                if len(line) <= 30:
                    continue
                # other_model = k[0].upper() + k[1:]
                # line = line.replace('Assistant 1', f"`{other_model}`").replace('Assistant 2', '`Vicuna`')
                new_lines.append(line)
            cleaned_evals[k] = '\n'.join(new_lines)

        r['evaluations'] = cleaned_evals
        records.append(r)

    data = {
        'questions': records,
        'models': models,
    }

    with open('data.json', 'w') as f:
        json.dump(data, f, indent=2)
