import json
import jsonlines
data_3 = "gen_data_v2_gpt3.5.jsonl"
data_4 = "gen_data_v2_gpt4.jsonl"
data_3q = "question.jsonl"
data_4q = "question4.jsonl"
data1 = "model_answer/gpt-3.5-turbo.jsonl"
data2 = "model_answer/gpt-3.5-turbo1.jsonl"
judge1 = "model_judgment/gpt-4_single1.jsonl"
judge2 = "model_judgment/gpt-4_single2.jsonl"
def convert(filename,output_filename):
    with open(filename,'r',encoding="utf-8") as rf:
        len = 1
        for item in rf.readlines():
            item = json.loads(item)
            q = {}
            q["question_id"] = len
            len += 1
            q["category"] = "QA"
            q["turns"] = []
            q["turns"].append(item["input"])
            with open(output_filename,'a',encoding="utf-8") as wf:
                json.dump(q,wf,ensure_ascii=False)
                wf.write('\n')

def change(filename,o_filename):
    with open(filename,'r',encoding="utf-8") as rf:
        len = 0
        for item in rf.readlines():
            item = json.loads(item)
            q = item
            q["question_id"] = str(len)
            len += 1
            with open(o_filename, 'a',encoding="utf-8") as wf:
                json.dump(q, wf, ensure_ascii=False)
                wf.write('\n')

def convert_code(f1,f2):
    with open(f1, 'r') as rf, open(f2, "a", encoding="utf-8") as wf:
        for item in rf.readlines():
            item = json.loads(item)
            json.dump(item, wf, ensure_ascii=False)
            wf.write('\n')
convert_code(judge1,judge2)
#change(data1,data2)