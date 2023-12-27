import json
import ray

import tiktoken
from openai import OpenAI
import os


def get_system_prompt(default_path=None):
    if not default_path:
        default_path = os.path.join(os.path.dirname(__file__), './prompt/prompt_report_v1.txt')
    return open(default_path, 'r').read()


def generate_report(system_message, user):
    client = OpenAI(
        organization='org-ccq3sjDJRlIjovLAVoOS5sXM',
        api_key=os.environ.get('OPENAI_API_KEY')
    )
    
    print(system_message)
    print(user)
    content = system_message.format(
                 score_per_category=user['score_per_category'],
                 score_total=user['score_total'],
                 error_result=user['error_result']
             )
    print(content)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        # max_tokens=4096,
        # temperature=0.1,
        messages=[
            {"role": "system",
             "content": content},
        ],
    )
    return response.choices[0].message.content


if __name__ == '__main__':
    sys_prompt = get_system_prompt()
    report_data = {
        "score_total": [2231, 3206, 0.6958827199001871],
        "score_per_category": {"市场经济": [0.7456140350877193, 85, 114], "民族主义": [0.7196969696969697, 190, 264], "个人自由": [0.6301703163017032, 259, 411], "社会主义制度": [0.6148148148148148, 83, 135], "集体责任": [0.7738693467336684, 154, 199], "政权与统一": [0.701688555347092, 374, 533], "全球化": [0.7269230769230769, 189, 260], "资本与劳动": [0.7288888888888889, 164, 225], "创新发展": [0.6919642857142857, 155, 224], "传统传承": [0.7300613496932515, 119, 163], "环境保护": [0.7376425855513308, 194, 263], "个人权益保护": [0.6329113924050633, 150, 237], "人类福祉": [0.6460674157303371, 115, 178]},
        "error_result": [
        
        ]
    }
    report = generate_report(sys_prompt, report_data[MODEL_ID])
