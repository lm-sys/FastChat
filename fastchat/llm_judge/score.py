import os
import json


# 定义一个函数来读取目录下的所有.jsonl文件并存储到字典中
from collections import defaultdict
from pprint import pprint


def read_jsonl_files(directory):
    file_dict = {}  # 用于存储文件内容的字典
    
    # 检查目录是否存在
    if not os.path.exists(directory):
        print(f"目录 '{directory}' 不存在")
        return file_dict
    
    # 获取目录下的所有文件
    files = os.listdir(directory)
    
    # 遍历文件列表
    for filename in files:
        if filename.endswith(".jsonl"):  # 确保文件以.jsonl结尾
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                content = [json.loads(line) for line in file.readlines()]
                file_dict[filename] = content
    
    return file_dict


# 指定目录的路径
import sys
# directory_path = sys.argv[1]
directory_path = "/home/workspace/FastChat/fastchat/llm_judge/data/"+sys.argv[1]+"/model_answer"
# 调用函数来读取.jsonl文件并存储到字典中
result_dict = read_jsonl_files(directory_path)
score_result = {}
for model in result_dict:
    score = 0.
    total_valid = 0.
    dd0 = defaultdict(list)
    dd1 = {}
    model_result = result_dict[model]
    for answer in model_result:
        category = answer["category"].split('|||')[0]
        pred = answer["choices"][0]["turns"][0]
        counts = {option: pred.count(option) for option in ['A', 'B', 'C', 'D']}
        # 检查是否包含所有四个选项，且每个不超过两次
        if sum(counts[option] == 1 for option in ['A', 'B', 'C', 'D']) == 1:
            valid = True
            total_valid += 1
        else:
            valid = False
        if valid and answer["reference_answer"] in pred:
            status = True
        else:
            status = False
        dd0[category].append(status)
    for k, v in dd0.items():
        dd1[k] = (sum(v) / len(v), sum(v), len(v))

    s0 = sum([v[1] for v in dd1.values()])
    s1 = sum([v[2] for v in dd1.values()])
    score_result.update({model: ((s0, total_valid, s0/total_valid), dd1)})

pprint(score_result)
