import json

cnt = 1
with open('temp_v2.jsonl', 'r') as f, open('question.jsonl', 'w') as g:
    for idx, line in enumerate(f):
        js = json.loads(line)
        topic = js['topic']
        policy = js['policy']
        for result in js['results']:
            try:
                q = result['question']
            except KeyError as e:
                print(e, result)
                continue
            id0 = result['id']
            options = '\n'.join(['%s:%s' % (k, v) for k, v in result['options'].items()])
            try:
                question_type = result['question_type']
            except KeyError as e:
                print(e, result)
                continue
            question_level = result['question_level']
            dd = {
                "question_id": cnt,
                "category": "%s|||%s" % (topic, policy),
                "turns": ["%s\n%s" % (q, options)],
                "reference_answer": result['reference_answer'],
                "question_type": result['question_type'],
                "question_level": result['question_level']
            }
            g.write(json.dumps(dd, ensure_ascii=False) + '\n')
            cnt += 1
        