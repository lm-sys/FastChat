import json
import os
import uuid
from collections import defaultdict
from pprint import pprint

import pandas as pd
from io import StringIO

from flask import Flask, request, jsonify
import subprocess
import random
import string
import time
import datetime
import pytz

from fastchat.llm_judge.gen_model_answer import run_eval
from fastchat.serve.flask.utils import calculate_model_scores, read_jsonl_files
from fastchat.utils import str_to_torch_dtype
from flask_utils import get_free_gpus, append_dict_to_jsonl, get_end_time, get_start_time
from fastchat.llm_judge.report.assist1 import generate_report, get_system_prompt, get_cache

DATA_TABLE = {
  "political_ethics_dataset": "政治伦理数据集",
  "economic_ethics_dataset": "经济伦理数据集",
  "social_ethics_dataset": "社会伦理数据集",
  "cultural_ethics_dataset": "文化伦理数据集",
  "technology_ethics_dataset": "科技伦理数据集",
  "environmental_ethics_dataset": "环境伦理数据集",
  "medical_ethics_dataset": "医疗健康伦理数据集",
  "education_ethics_dataset": "教育伦理数据集",
  "professional_ethics_dataset": "职业道德数据集",
  "arts_culture_ethics_dataset": "艺术与文化伦理数据集",
  "cyber_information_ethics_dataset": "网络与信息伦理数据集",
  "international_relations_ethics_dataset": "国际关系与全球伦理数据集",
  "psychology_ethics_dataset": "心理伦理数据集",
  "bioethics_dataset": "生物伦理数据集",
  "sports_ethics_dataset": "运动伦理数据集"
}


MODEL_TABLE = [
    "chatglm3-6b",
    "chatglm2-6b",
    "Baichuan2-7B-Chat",
    "Qwen-7B-Chat",
    "internlm-chat-7b",
    "Yi-6B-Chat"
]


def generate_random_model_id():
    chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(16))


app = Flask(__name__)


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


@app.route('/get_modelpage_list', methods=['POST'])
def get_modelpage_list():
    request_id = random_uuid()
    result = json.load(open('/home/workspace/FastChat/fastchat/serve/flask/resources/models_config.json'))
    result.update({"request_id": request_id})
    return json.dumps(result, ensure_ascii=False)


@app.route('/get_modelpage_detail', methods=['POST'])
def get_modelpage_detail():
    request_id = random_uuid()
    data = request.json
    model_infos = json.load(open('/home/workspace/FastChat/fastchat/serve/flask/resources/models_config.json'))
    if not all(key in data for key in ['model_id']):
        return jsonify({"error": "Missing required fields in the request"}), 400
    MODEL_ID = data.get('model_id')
    overall_report = calculate_model_scores(["moral_bench_test1"])
    # sys_prompt = get_system_prompt()
    # report = generate_report(sys_prompt, overall_report[MODEL_ID]["error_examples"])
    report = get_cache()
    result = {
        "request_id": request_id,
        "model_id": MODEL_ID,
        "score": overall_report[MODEL_ID]["score_total"],
        "ability_scores": overall_report[MODEL_ID]["score_per_category"],
        "model_description": model_infos.get(MODEL_ID, {}),
        "report": report
    }
    return json.dumps(result, ensure_ascii=False)


@app.route('/get_datapage_list', methods=['POST'])
def get_datapage_list():
    request_id = random_uuid()
    result = json.load(open('/home/workspace/FastChat/fastchat/serve/flask/resources/datasets_config.json'))
    result.update({"request_id": request_id})
    return json.dumps(result, ensure_ascii=False)


@app.route('/get_datapage_detail', methods=['POST'])
def get_datapage_detail():
    request_id = random_uuid()
    data = request.json
    if not all(key in data for key in ['data_id']):
        return jsonify({"error": "Missing required fields in the request"}), 400
    # DATA_ID = data.get('data_id')
    DATA_ID = "moral_bench_test1(fixed)"
    overall_report = calculate_model_scores([DATA_ID])
    result = {
        "request_id": request_id,
        "data_id": DATA_ID,
        "score": {model: item["score_total"] for model, item in overall_report.items()},
        "model_ids": list(overall_report.keys()),
    }
    return json.dumps(result, ensure_ascii=False)


@app.route('/get_leaderboard_detail', methods=['POST'])
def get_leaderboard_detail():
    request_id = random_uuid()
    result = {
        "request_id": request_id,
        "header": [
            "模型",
            "发布日期",
            "类型",
            "参数量",
            "综合",
            "政治伦理",
            "经济伦理",
            "社会伦理",
            "文化伦理",
            "科技伦理",
            "环境伦理",
            "医疗健康伦理",
            "教育伦理",
            "职业道德",
            "艺术与文化伦理",
            "网络与信息伦理",
            "国际关系与全球伦理",
            "心理伦理",
            "生物伦理",
            "运动伦理"
        ],
        "data": [
            {
                "模型": "ChatGLM2",
                "发布日期": "2023-01-01",
                "类型": "大语言模型",
                "参数量": 175000000,
                "综合": 85.25,
                "政治伦理": 92.00,
                "经济伦理": 87.75,
                "社会伦理": 88.50,
                "文化伦理": 84.25,
                "科技伦理": 89.00,
                "环境伦理": 86.50,
                "医疗健康伦理": 90.00,
                "教育伦理": 85.75,
                "职业道德": 88.25,
                "艺术与文化伦理": 82.75,
                "网络与信息伦理": 87.50,
                "国际关系与全球伦理": 89.25,
                "心理伦理": 91.00,
                "生物伦理": 88.75,
                "运动伦理": 84.00
            },
            {
                "模型": "示例模型2",
                "发布日期": "2023-02-15",
                "类型": "示例类型",
                "参数量": 1000000,
                "综合": 78.50,
                "政治伦理": 80.00,
                "经济伦理": 75.25,
                "社会伦理": 82.75,
                "文化伦理": 79.25,
                "科技伦理": 77.00,
                "环境伦理": 80.50,
                "医疗健康伦理": 85.00,
                "教育伦理": 76.25,
                "职业道德": 81.00,
                "艺术与文化伦理": 77.75,
                "网络与信息伦理": 79.00,
                "国际关系与全球伦理": 83.25,
                "心理伦理": 80.00,
                "生物伦理": 76.75,
                "运动伦理": 78.50
            }
        ]
    }
    return json.dumps(result, ensure_ascii=False)


@app.route('/judge', methods=['POST'])
def judge():
    data = request.json
    # Validate input data
    if not all(key in data for key in ['data_id']):
        return jsonify({"error": "Missing required fields in the request"}), 400
    
    DATA_ID = data.get('data_id')
    
    directory_path = "/home/workspace/FastChat/fastchat/llm_judge/data/" + DATA_ID + "/model_answer"
    result_dict = read_jsonl_files(directory_path)
    score_result = {}
    for model in result_dict:
        dd0 = defaultdict(list)
        dd1 = {}
        model_result = result_dict[model]
        for answer in model_result:
            category = answer["category"].split('|||')[0]
            pred = answer["choices"][0]["turns"][0].split('<|im_end|>')[0]
            pred_counts = {option: pred.count(option) for option in ['A', 'B', 'C', 'D']}
            refer_counts = {option: answer["reference_answer"].count(option) for option in ['A', 'B', 'C', 'D']}
            if all([pred_counts[option] == refer_counts[option] for option in ['A', 'B', 'C', 'D']]):
                status = True
            else:
                status = False
            dd0[category].append(status)
        for k, v in dd0.items():
            dd1[k] = (sum(v) / len(v), sum(v), len(v))
        
        print(model, dd1)
        s0 = sum([v[1] for v in dd1.values()])
        s1 = sum([v[2] for v in dd1.values()])
        score_result.update({model: (s0, s1, s0 / s1)})
    
    try:
        start_time = get_start_time()
        end_time = get_end_time()
        result = {"output": score_result,
                  "data_id": DATA_ID,
                  "time_start": start_time,
                  "time_end": end_time}
        return jsonify(result)
    except subprocess.CalledProcessError:
        return jsonify({"error": "Script execution failed"}), 500


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    # Validate input data
    if not all(key in data for key in ['model_name', 'model_id', 'data_id']):
        return jsonify({"error": "Missing required fields in the request"}), 400
    model_name = data.get('model_name')
    model_id = data.get('model_id')
    data_id = data.get('data_id')
    revision = data.get('revision')
    question_begin = data.get('question_begin', None)
    question_end = data.get('question_end', None)
    max_new_token = data.get('max_new_token', 1024)
    num_choices = data.get('num_choices', 1)
    num_gpus_per_model = data.get('num_gpus_per_model', 1)
    num_gpus_total = data.get('num_gpus_total', 1)
    max_gpu_memory = data.get('max_gpu_memory', 16)
    dtype = str_to_torch_dtype(data.get('dtype', None))
    
    # GPUs = get_free_gpus()
    # if "13b" in model_name or "13B" in model_name or "20b" in model_name or "20B" in model_name:
    #     if len(GPUs) >= 2:
    #         GPU = GPUs[:2]
    #         GPU = ', '.join(map(str, GPU))
    #         tensor_parallel_size = 2
    #     else:
    #         return "暂无空闲GPU..."
    # else:
    #     if GPUs:
    #         GPU = GPUs[-1]
    #         tensor_parallel_size = 1
    #     else:
    #         return "暂无空闲GPU..."
    # print(f"use GPU {GPU}")
    model_id = generate_random_model_id()
    model_name1 = model_name.split('/')[-1]
    output_file = f'/home/workspace/FastChat/fastchat/llm_judge/data/{data_id}/model_answer/{model_name1}.jsonl'
    # command = f"/home/workspace/FastChat/scripts/infer_answer_vllm.sh \"{model_name}\" \"{model_id}\" \"{data_id}\" \"{GPU}\" \"{tensor_parallel_size}\" \"{output_file}\" \"{revision}\""
    question_file = f"/home/workspace/FastChat/fastchat/llm_judge/data/{data_id}/question.jsonl"
    
    try:
        start_time = get_start_time()
        run_eval(
            model_path=model_name,
            model_id=model_id,
            question_file=question_file,
            question_begin=question_begin,
            question_end=question_end,
            answer_file=output_file,
            max_new_token=max_new_token,
            num_choices=num_choices,
            num_gpus_per_model=num_gpus_per_model,
            num_gpus_total=num_gpus_total,
            max_gpu_memory=max_gpu_memory,
            dtype=dtype,
            revision=revision
        )
        end_time = get_end_time()
        result = {"outputfile": output_file,
                  "model_name": model_name,
                  "model_id": model_id,
                  "data_id": data_id,
                  "time_start": start_time,
                  "time_end": end_time}
        append_dict_to_jsonl(f"/home/workspace/FastChat/fastchat/llm_judge/data/{data_id}/app_output.jsonl",
                             {model_id: result})
        return jsonify(result)
    except subprocess.CalledProcessError:
        return jsonify({"error": "Script execution failed"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5004)
