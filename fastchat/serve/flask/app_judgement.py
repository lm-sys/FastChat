from flask import Flask, request, jsonify
import subprocess
import random
import string
import time
import datetime
import pytz
from flask_utils import get_free_gpus, generate_random_identifier,get_end_time,get_start_time




def generate_random_identifier():
    chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(16))


app = Flask(__name__)


@app.route('/judgement', methods=['POST'])
def run_script():
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

    output_name=JUDGE_MODEL
    ANSWER_LIST= ANSWER_LIST.split()
    for answer in ANSWER_LIST:
        output_name+=("_"+answer)
    output_file = f'/home/Userlist/madehua/code/fc/fastchat/llm_judge/data/single_turn/model_judgment/{output_name}_{MODE}.jsonl'


    return jsonify({"outputfile": output_file,
                    "time_start": start_time,
                    "time_end": end_time}
                   )



if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True,port=5001)
