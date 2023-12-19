import json

with open('temp_v10.jsonl', 'r') as f, open('question.jsonl', 'w') as g:
    for idx, line in enumerate(f):
        js = json.loads(line)
        js['question_id'] = idx + 1
        g.write(json.dumps(js, ensure_ascii=False) + '\n')