import os
import time
import json
import argparse
import pandas as pd
from tqdm import tqdm
from gradio_client import Client
from typing import Dict

from threading import Lock
from concurrent.futures import ThreadPoolExecutor

def sample(prompt: str, model_name: str) -> str:
    GRADIO_CLIENT = Client("http://0.0.0.0:7860")
    GRADIO_CLIENT.predict(
        model_selector=model_name,
        text=prompt,
        api_name="/add_text_1")
    response = GRADIO_CLIENT.predict(api_name="/bot_response_2")[0][1]
    return response


def process_row(row: pd.Series, model_name: str, input_field_name: str) -> Dict:
    prompt = row[input_field_name]
    question_id = row["question_id"]
    response = sample(prompt, model_name)
    while True:
        response = sample(prompt, model_name)
        if "Status code 429. Rate limit exceeded." in response and "firecrawl.dev" in response:
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
        output_path = "eval_outputs/{dataset_name}_{model_name}.jsonl".format(dataset_name=dataset_name, model_name=args.model_name)
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
                        process_row(row[1], args.model_name, args.input_field_name),
                        lock),
                    rows),
                total=len(rows)
            )
       )
    
if __name__ == "__main__":
    main()