import re
import os
import json
import random
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
random.seed(42)


import prompts

class ArenaHardGrader():
    GRADER_PROMTP = prompts.LLM_JUDGE_PROMPT
    def __init__(self, grader_model_name: str = "gpt-4o"):
        self.grader_model_name = grader_model_name
    
    def grader_model(self, prompt_messages: List[Dict]) -> str:
            completion = OpenAI_CLIENT.chat.completions.create(
                model=self.grader_model_name,
                messages=prompt_messages,
                temperature=0.0,
                top_p=1.0,
            )
            return completion.choices[0].message.content
    
    def grade_sample(self, question: str, base_answer: str, agent_answer: str):
        if random.random() > 0.5:
            answer_a = base_answer
            answer_b = agent_answer
            agent_answer_order = "B"
            base_answer_order = "A"
        else:
            answer_a = agent_answer
            answer_b = base_answer
            agent_answer_order = "A"
            base_answer_order = "B"
        grader_prompt = self.GRADER_PROMTP.format(
            question=question,
            answer_a=answer_a,
            answer_b=answer_b,
        )
        prompt_messages = [
            {"role": "user", "content": grader_prompt}
        ]
        grading_response = self.grader_model(prompt_messages)
        match = re.search(r"\[\[(A|B|C)\]\]", grading_response)
        return_str = f"Base: {base_answer_order}; Agent: {agent_answer_order}\n\nGrading Response: {grading_response}"
        verdict = match.group(0) if match else "C"
        if "C" in verdict:
             return "tie", return_str
        elif agent_answer_order in verdict:
            return "agent", return_str
        return "base", return_str

def write_to_file_safe(output_path, data, lock):
    with lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(data) + "\n")


def process_row(row: pd.Series):
    question = row["prompt"]
    base_answer = row["base_response"]
    agent_answer = row["agent_response"]
    question_id = row["question_id"]
    do_search = row["search_done_y"]
    grader = ArenaHardGrader()
    grade, raw_response = grader.grade_sample(question, base_answer, agent_answer)
    return {"question_id": question_id, "question": question, "grade": grade, "base_response": base_answer, "agent_response": agent_answer, "do_search": do_search, "raw_response": raw_response}

def main():
    parser = argparse.ArgumentParser(description="Process evaluation arguments.")
    parser.add_argument("--eval_inputs_path", type=str, required=True, help="File path for the input evaluation dataset.")
    parser.add_argument("--sample_outputs_base_path", type=str, required=True, help="File path for the sample outputs of the base model.")
    parser.add_argument("--sample_outputs_agent_path", type=str, required=True, help="File path for the sample outputs of the agent model.")
    parser.add_argument("--output_path", type=str, help="Path to save the generated responses.")
    parser.add_argument("--grader_model_name", type=str, default="gpt-4o", help="Model used to evaluate.")
    args = parser.parse_args()
    
    input_data = pd.read_json(args.eval_inputs_path, lines=True)
    base_model_samples = pd.read_json(args.sample_outputs_base_path, lines=True)
    base_model_samples.rename(columns={"response": "base_response"}, inplace=True)
    agent_model_samples = pd.read_json(args.sample_outputs_agent_path, lines=True)
    agent_model_samples.rename(columns={"response": "agent_response"}, inplace=True)
    if "question_id" not in input_data.columns:
        raise ValueError("question_id not found in the input dataset.")
    if "question_id" not in base_model_samples.columns:
        raise ValueError("question_id not found in the base model samples dataset.")
    if "question_id" not in agent_model_samples.columns:
        raise ValueError("question_id not found in the agent model samples dataset.")
    
    data = pd.merge(input_data, base_model_samples, on="question_id", how="inner")
    data = pd.merge(data, agent_model_samples, on="question_id", how="inner")
    output_path = args.output_path
    with open(output_path, "w") as f:
        pass
    lock = Lock()
    
    rows = list(data.iterrows())
    with ThreadPoolExecutor(max_workers=4) as executor:
       _ = list(tqdm(
                executor.map(
                    lambda row: write_to_file_safe(
                        output_path,
                        process_row(row[1]),
                        lock),
                    rows),
                total=len(rows)
            )
       )
    
if __name__ == "__main__":
    main()