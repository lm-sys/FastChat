import json
import os
from collections import defaultdict

from flask import Flask, request, jsonify
import subprocess
import random
import string
import time
import datetime
import pytz
from flask_utils import get_free_gpus, generate_random_identifier, append_dict_to_jsonl, get_end_time, get_start_time


def generate_random_identifier():
    chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(16))


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

app = Flask(__name__)


@app.route('/judge', methods=['POST'])
def run_script_judgement():
    data = request.json
    # Validate input data
    if not all(key in data for key in ['judgement_model', 'mode', 'answer_list']):
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
def run_script_generate():
    data = request.json
    # Validate input data
    if not all(key in data for key in ['model_name', 'model_id', 'data_id']):
        return jsonify({"error": "Missing required fields in the request"}), 400
    model_name = data.get('model_name')
    model_id = data.get('model_id')
    data_id = data.get('data_id')
    GPUs = get_free_gpus()
    if "13b" in model_name or "13B" in model_name or "20b" in model_name or "20B" in model_name:
        if len(GPUs) >= 2:
            GPU = GPUs[:2]
            GPU = ', '.join(map(str, GPU))
            
            tensor_parallel_size = 2
        else:
            return "暂无空闲GPU..."
    else:
        if GPUs:
            GPU = GPUs[-1]
            tensor_parallel_size = 1
        else:
            return "暂无空闲GPU..."
    print(f"use GPU {GPU}")
    identifier = generate_random_identifier()
    model_name1 = model_name.split('/')[-1]
    output_file = f'/home/workspace/FastChat/fastchat/llm_judge/data/{data_id}/model_answer/{model_name1}.jsonl'
    command = f"/home/workspace/FastChat/scripts/infer_answer_vllm.sh \"{model_name}\" \"{model_id}\" \"{data_id}\" \"{GPU}\" \"{tensor_parallel_size}\" \"{output_file}\""
    
    try:
        start_time = get_start_time()
        subprocess.check_call(command, shell=True)
        end_time = get_end_time()
        result = {"outputfile": output_file,
                  "model_name": model_name,
                  "model_id": model_id,
                  "data_id": data_id,
                  "time_start": start_time,
                  "time_end": end_time}
        append_dict_to_jsonl(f"/home/workspace/FastChat/fastchat/llm_judge/data/{data_id}/app_output.jsonl", {identifier: result})
        return jsonify(result)
    except subprocess.CalledProcessError:
        return jsonify({"error": "Script execution failed"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5004)
