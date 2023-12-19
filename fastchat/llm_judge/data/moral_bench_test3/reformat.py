import json

cnt = 1
with open('temp_v10.jsonl', 'r') as f, open('question.jsonl', 'w') as g:
    for idx, line in enumerate(f):
        js = json.loads(line)
        tt = js['turns']
        rr = js['reference_answer']
        if len(tt) == len(rr):
            for t, r in zip(tt, rr):
                js['question_id'] = cnt
                cnt += 1
                js['turns'] = [t]
                js['reference_answer'] = [r]
                g.write(json.dumps(js, ensure_ascii=False) + '\n')
        else:
            print(idx, line)