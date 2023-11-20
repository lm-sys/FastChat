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


app = Flask(__name__)


@app.route('/judgement', methods=['POST'])
def run_script_judgement():
    data = request.json
    # Validate input data
    if not all(key in data for key in ['judgement_model', 'mode', 'answer_list']):
        return jsonify({"error": "Missing required fields in the request"}), 400
    
    JUDGE_MODEL = data.get('judgement_model')
    MODE = data.get('mode')
    DATA_ID = data.get('data_id')
    ANSWER_LIST = data.get('answer_list')
    
    command = f"sh /home/Userlist/madehua/code/fc/fastchat/llm_judge/judgement.sh \"{JUDGE_MODEL}\" \"{MODE}\" \"{DATA_ID}\" \"{ANSWER_LIST}\" "
    
    start_time = get_start_time()
    subprocess.check_call(command, shell=True)
    end_time = get_end_time()
    
    output_name = JUDGE_MODEL
    ANSWER_LIST = ANSWER_LIST.split()
    for answer in ANSWER_LIST:
        output_name += ("_" + answer)
    output_file = f'/home/Userlist/madehua/code/fc/fastchat/llm_judge/data/single_turn/model_judgment/{output_name}_{MODE}.jsonl'
    
    return jsonify({"outputfile": output_file,
                    "time_start": start_time,
                    "time_end": end_time}
                   )


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
    command = f"/root/autodl-tmp/software/FastChat/scripts/infer_answer_vllm.sh \"{model_name}\" \"{model_id}\" \"{data_id}\" \"{GPU}\" \"{tensor_parallel_size}\" \"{identifier}\""
    
    try:
        start_time = get_start_time()
        subprocess.check_call(command, shell=True)
        end_time = get_end_time()
        
        output_file = f'/root/autodl-tmp/software/FastChat/fastchat/llm_judge/data/moral_bench/model_answer/{model_id}.jsonl'
        result = {"outputfile": output_file,
                  "model_name": model_name,
                  "model_id": model_id,
                  "data_id": data_id,
                  "time_start": start_time,
                  "time_end": end_time}
        append_dict_to_jsonl("/root/autodl-tmp/software/FastChat/fastchat/llm_judge/data/moral_bench/model_answer/app_output.jsonl", {identifier: result})
        return jsonify(result)
    except subprocess.CalledProcessError:
        return jsonify({"error": "Script execution failed"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5004)
