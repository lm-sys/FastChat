import re
import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from typing import List, Dict

from threading import Lock
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv('keys.env')

from openai import OpenAI
OpenAI_CLIENT = OpenAI()

import prompts

class SimpleQAGrader():
    GRADER_PROMPT = prompts.SIMPLE_QA_GRADER_PROMPT
    def __init__(self, grader_model_name: str = "gpt-4o"):
        self.grader_model_name = grader_model_name
    
    def grader_model(self, prompt_messages: List[Dict]) -> str:
        completion = OpenAI_CLIENT.chat.completions.create(
            model=self.grader_model_name,
            messages=prompt_messages
        )
        return completion.choices[0].message.content

    def grade_sample(self, question: str, target: str, predicted_answer: str) -> str:
        grader_prompt = self.GRADER_PROMPT.format(
            question=question,
            target=target,
            predicted_answer=predicted_answer,
        )
        prompt_messages = [
            {"role": "user", "content": grader_prompt}
        ]
        grading_response = self.grader_model(prompt_messages)
        match = re.search(r"(A|B|C)", grading_response)
        return match.group(0) if match else "C"

def write_to_file_safe(output_path, data, lock):
    with lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(data) + "\n")

def process_row(row: pd.Series, prompt_field_name: str, response_field_name: str, target_field_name: str, grader_model_name: str):
    question = row[prompt_field_name]
    target = row[target_field_name]
    predicted_answer = row[response_field_name]
    question_id = row["question_id"]
    grader = SimpleQAGrader(grader_model_name=grader_model_name)
    grade = grader.grade_sample(question, target, predicted_answer)
    return {"question_id": question_id, "grade": grade}


def main():
    parser = argparse.ArgumentParser(description="Process evaluation arguments.")
    parser.add_argument("--eval_inputs_path", type=str, required=True, help="File path for the input evaluation dataset.")
    parser.add_argument("--sample_outputs_path", type=str, required=True, help="File path for the sample outputs.")
    parser.add_argument("--grader_model_name", type=str, default="gpt-4o", help="Model used to evaluate.")
    args = parser.parse_args()

    
    input_data = pd.read_json(args.eval_inputs_path, lines=True)
    model_samples = pd.read_json(args.sample_outputs_path, lines=True)
    if "question_id" not in input_data.columns:
        raise ValueError("question_id not found in the input dataset.")
    if "question_id" not in model_samples.columns:
        raise ValueError("question_id not found in the sample outputs dataset.")
    
    data = pd.merge(input_data, model_samples, on="question_id")
    dataset_model_name = os.path.splitext(os.path.basename(args.sample_outputs_path))[0]
    output_path = "eval_results/{}_results.jsonl".format(dataset_model_name)
    with open(output_path, "w") as f:
        pass
    lock = Lock()
    
    rows = list(data.iterrows())
    with ThreadPoolExecutor(max_workers=4) as executor:
       _ = list(tqdm(
                executor.map(
                    lambda row: write_to_file_safe(
                        output_path,
                        process_row(row[1], "prompt", "response", "label", args.grader_model_name),
                        lock),
                    rows),
                total=len(rows)
            )
       )
    
if __name__ == "__main__":
    main()