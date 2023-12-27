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