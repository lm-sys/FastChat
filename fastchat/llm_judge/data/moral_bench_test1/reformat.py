import json

with open('question.jsonl', 'r') as f, open('question1.jsonl', 'w') as g:
    for idx, line in enumerate(f):
        js = json.loads(line)
        js['question_id'] = idx + 1
        g.write(json.dumps(js, ensure_ascii=False) + '\n')