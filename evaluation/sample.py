import os
import time
import json
import requests
import argparse
import pandas as pd
from tqdm import tqdm
from gradio_client import Client
from typing import Dict

from threading import Lock
from concurrent.futures import ThreadPoolExecutor

os.environ["PERPLEXITY_API_KEY"] = ""

def sample_gradio(prompt: str, model_name: str) -> str:
    GRADIO_CLIENT = Client("http://0.0.0.0:7860")
    GRADIO_CLIENT.predict(
        model_selector=model_name,
        text=prompt,
        api_name="/add_text_1")
    response = GRADIO_CLIENT.predict(api_name="/bot_response_2")[0][1]
    return response

def sample_perplexity(prompt: str, model_name: str) -> str:
    PPL_URL = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "search_domain_filter": ["perplexity.ai"],
        "return_images": False,
        "return_related_questions": False,
        "search_recency_filter": "month",
        "top_k": 0,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1
    }
    headers = {
        "Authorization": "Bearer " + os.getenv("PERPLEXITY_API_KEY"),
        "Content-Type": "application/json"
    }
    while True:
        response = requests.request("POST", PPL_URL, json=payload, headers=headers).json()
        if "choices" not in response.keys():
            print(response)
            time.sleep(65)
        else:
            break
    response_text = response['choices'][0]['message']['content']
    if "citations" in response.keys() and response["citations"]:
        response_text = response_text + "\nReference Website: \n\n- " + "\n- ".join(response["citations"])
    return response_text


def process_row(row: pd.Series, model_type: str, model_name: str, input_field_name: str) -> Dict:
    prompt = row[input_field_name]
    question_id = row["question_id"]
    if model_type == "gradio":
        sample = sample_gradio
    elif model_type == "perplexity":
        sample = sample_perplexity
    response = sample(prompt, model_name)
    while True:
        response = sample(prompt, model_name)
        if "rate limit exceeded" in response.lower():
            time.sleep(65)
        else:
            break
    search_done = "reference website" in response.lower()
    return {"question_id": question_id, "response": response, "search_done": search_done}


def write_to_file_safe(output_path, data, lock):
    with lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(data) + "\n")
    

def main():
    parser = argparse.ArgumentParser(description="Process evaluation arguments.")
    parser.add_argument("--model_type", type=str, required=True, help="Model API type.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name to evaluate.")
    parser.add_argument("--dataset_path", type=str, required=True, help="File path for the evaluation dataset.")
    parser.add_argument("--input_field_name", type=str, default="prompt", help="Name of the input field going to the LLM agent.")
    parser.add_argument("--output_path", type=str, help="Path to save the generated responses.")
    args = parser.parse_args()
    
    dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
    input_data = pd.read_json(args.dataset_path, lines=True)
    if args.input_field_name not in input_data.columns:
        raise ValueError(f"Input field name {args.input_field_name} not found in the dataset.")
    if "question_id" not in input_data.columns:
        raise ValueError("question_id not found in the dataset.")
    
    if not args.output_path:
        output_path = "eval_samples/{dataset_name}_{model_name}.jsonl".format(dataset_name=dataset_name, model_name=args.model_name)
    else:
        output_path = args.output_path
    with open(output_path, "w") as f:
        pass
    lock = Lock()
    
    rows = list(input_data.iterrows())
    with ThreadPoolExecutor(max_workers=4) as executor:
       _ = list(tqdm(
                executor.map(
                    lambda row: write_to_file_safe(
                        output_path,
                        process_row(row[1], args.model_type, args.model_name, args.input_field_name),
                        lock),
                    rows),
                total=len(rows)
            )
       )
    
if __name__ == "__main__":
    main()