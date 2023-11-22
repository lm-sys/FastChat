import os
import json


# 定义一个函数来读取目录下的所有.jsonl文件并存储到字典中
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
directory_path = sys.argv[1]

# 调用函数来读取.jsonl文件并存储到字典中
result_dict = read_jsonl_files(directory_path)
score_result = {}
for model in result_dict:
    score = 0
    zeroCount = 0
    model_result = result_dict[model]
    for answer in model_result:
        pred = answer["choices"][0]["turns"][0]
        count = 0
        if "A" in pred:
            count += 1
        if "B" in pred:
            count += 1
        if "C" in pred:
            count += 1
        if "D" in pred:
            count += 1
        if answer["reference_answer"] in pred and count == 1:
            score += 1
        if count != 0:
            zeroCount += 1
    score_result.update({model: [score, zeroCount]})

print(score_result)
