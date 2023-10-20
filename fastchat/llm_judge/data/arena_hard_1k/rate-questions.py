import pandas as pd
import glob
import os
import re
from tqdm import tqdm

path = os.path.join("model_answer", "*.jsonl")
files = glob.glob(path)

questions = pd.read_json("question.jsonl", lines=True)
questions["mean_score"] = None


pattern = re.compile(r"\[\[(.*?)\]\]")

def get_score(text):
    matches = pattern.findall(text)

    if len(matches) == 1:
        return float(matches[0])
    

# begin
answers = pd.read_json(os.path.join("model_judgment", "gpt-4_single.jsonl"), lines=True)
models = []
for file in files:
    models.append(file.replace(".jsonl", "").replace("model_answer/", ""))

for q, question in tqdm(questions.iterrows()):
    id = question["question_id"]

    total_score = 0
    answer = answers[answers["question_id"] == id]
    for model in models:
        judgment = answer[answer["model"] == model].iloc[0]["judgment"]
        score = get_score(judgment)

        if not score:
            continue
        
        total_score += score
    
    questions.at[q, "mean_score"] = int(total_score / len(models))

questions.to_json("rated_question.jsonl", orient="records", lines=True)


    