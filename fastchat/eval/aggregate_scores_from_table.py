"""Aggregate and print out scores."""
import json
import os
import re


def read_jsonl(path: str, key: str = None):
    data = []
    with open(os.path.expanduser(path)) as f:
        for line in f:
            if not line:
                continue
            data.append(json.loads(line))
    if key is not None:
        data.sort(key=lambda x: x[key])
        data = {item[key]: item for item in data}
    return data



if __name__ == "__main__":

    #REVIEW_BASE = ""
    REVIEW_BASE = "gpt-3.5-turbo"
    #MODEL_BASE = "vicuna-13b-20230322-clean-lang"
    MODEL_BASE = "vicuna-13b-20230322-new-hp-fp16"
    #MODEL_BASE = "vicuna-7b-20230322-fp16"

    review_filenames = os.listdir(f"table/review/{REVIEW_BASE}_review_{MODEL_BASE}")
    #review_filenames = os.listdir(f"table/review/{MODEL_BASE}")

    results = {}
    for review_filename in review_filenames:
        #if not review_filename.startswith(REVIEW_BASE): continue
        if not review_filename.endswith(f"{MODEL_BASE}.jsonl"): continue
        model_substring = review_filename[len(f"{REVIEW_BASE}_review_"):]
        #model_substring = review_filename[len(f"review_"):]
        model_name = model_substring.split("_")[0]
        print(f"Loading review {review_filename} for model {model_name}")
        review = read_jsonl(f"table/review/{REVIEW_BASE}_review_{MODEL_BASE}/{review_filename}", key="question_id")
        #review = read_jsonl(f"table/review/{MODEL_BASE}/{review_filename}", key="question_id")
        total_score1 = 0
        total_score2 = 0
        failed_eval_count = 0
        better1 = 0
        better2 = 0
        tie = 0
        for v in review.values():
            score1, score2 = v["score"]
            if score1 == -1 or score2 == -1:
                failed_eval_count += 1
                continue
            total_score1 += score1
            total_score2 += score2
            if score1 > score2: better1 += 1
            elif score2 > score1: better2 += 1
            else: tie += 1
        results[model_name] = {
                "total_score1": total_score1,
                "total_score2": total_score2,
                "failed_eval_count": failed_eval_count,
                "better1": better1,
                "better2": better2,
                "tie": tie,
                }

    print(json.dumps(results, indent=2))
