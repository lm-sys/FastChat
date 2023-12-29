import json
import os
import string
import subprocess
from collections import defaultdict
from pprint import pprint
from random import random


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
                file_dict[filename.split('.jsonl')[0]] = content
    
    return file_dict


from collections import defaultdict

from collections import defaultdict


def calculate_model_scores(data_id_list):
    overall_report = {}
    error_results = []
    
    for data_id in data_id_list:
        answers_directory_path = f"/home/workspace/FastChat/fastchat/llm_judge/data/{data_id}/model_answer"
        model_answers = read_jsonl_files(answers_directory_path)
        
        for model, answers in model_answers.items():
            if model not in overall_report:
                overall_report[model] = {"total_correct": 0, "total_questions": 0,
                                         "score_per_category": defaultdict(lambda: {"correct": 0, "total": 0})}
            
            for answer in answers:
                if len(answer["reference_answer"]) > 1:
                    print("invalid reference answer", answer)
                    continue
                category = answer["category"].split('|||')[0]
                predicted = answer["choices"][0]["turns"][0].strip()
                predicted_counts = {option: predicted.count(option) for option in ['A', 'B', 'C', 'D']}
                reference_counts = {option: answer["reference_answer"].count(option) for option in ['A', 'B', 'C', 'D']}
                is_correct = all(predicted_counts[opt] == reference_counts[opt] for opt in ['A', 'B', 'C', 'D'])
                
                if not is_correct:
                    error_results.append({
                        "category": category,
                        "predicted": [k for k, v in predicted_counts.items() if v > 0],
                        "reference": [k for k, v in reference_counts.items() if v > 0],
                        "question": answer["question"].split("仅输出选项A、B、C、D中的一个即可:")[-1],
                    })
                
                overall_report[model]["score_per_category"][category]["correct"] += is_correct
                overall_report[model]["score_per_category"][category]["total"] += 1
                overall_report[model]["total_correct"] += is_correct
                overall_report[model]["total_questions"] += 1
    
    # Finalize the report
    for model, data in overall_report.items():
        for category, scores in data["score_per_category"].items():
            data["score_per_category"][category] = {
                "correct": scores["correct"],
                "total": scores["total"],
                "accuracy": scores["correct"] / scores["total"]
            }
        
        data["score_total"] = data["total_correct"] / data["total_questions"]
        data["error_examples"] = error_results[:3]
    
    return overall_report


def generate_random_identifier():
    chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(16))


def get_free_gpus():
    try:
        # 执行 nvidia-smi 命令
        cmd = "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode("utf-8")
        
        # 分析输出结果
        free_gpus = []
        lines = output.strip().split("\n")
        for line in lines:
            index, memory_used = line.split(", ")
            if int(memory_used) <= 100:
                free_gpus.append(int(index))
        
        return free_gpus
    except Exception as e:
        print(f"Error: {e}")
        return []
