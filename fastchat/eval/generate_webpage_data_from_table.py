"""Generate json file for webpage."""
import json
import os
import re

models = ["alpaca", "llama", "gpt35", "bard"]


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


def trim_hanging_lines(s: str, n: int) -> str:
    s = s.strip()
    for _ in range(n):
        s = s.split("\n", 1)[1].strip()
    return s


if __name__ == "__main__":
    questions = read_jsonl("table/question.jsonl", key="question_id")

    alpaca_answers = read_jsonl(
        "table/answer/answer_alpaca-13b.jsonl", key="question_id"
    )
    bard_answers = read_jsonl("table/answer/answer_bard.jsonl", key="question_id")
    gpt35_answers = read_jsonl("table/answer/answer_gpt35.jsonl", key="question_id")
    llama_answers = read_jsonl("table/answer/answer_llama-13b.jsonl", key="question_id")
    vicuna_answers = read_jsonl(
        "table/answer/answer_vicuna-13b.jsonl", key="question_id"
    )

    review_alpaca = read_jsonl(
        "table/review/review_alpaca-13b_vicuna-13b.jsonl", key="question_id"
    )
    review_bard = read_jsonl(
        "table/review/review_bard_vicuna-13b.jsonl", key="question_id"
    )
    review_gpt35 = read_jsonl(
        "table/review/review_gpt35_vicuna-13b.jsonl", key="question_id"
    )
    review_llama = read_jsonl(
        "table/review/review_llama-13b_vicuna-13b.jsonl", key="question_id"
    )

    records = []
    for qid in questions.keys():
        r = {
            "id": qid,
            "category": questions[qid]["category"],
            "question": questions[qid]["text"],
            "answers": {
                "alpaca": alpaca_answers[qid]["text"],
                "llama": llama_answers[qid]["text"],
                "bard": bard_answers[qid]["text"],
                "gpt35": gpt35_answers[qid]["text"],
                "vicuna": vicuna_answers[qid]["text"],
            },
            "evaluations": {
                "alpaca": review_alpaca[qid]["text"],
                "llama": review_llama[qid]["text"],
                "bard": review_bard[qid]["text"],
                "gpt35": review_gpt35[qid]["text"],
            },
            "scores": {
                "alpaca": review_alpaca[qid]["score"],
                "llama": review_llama[qid]["score"],
                "bard": review_bard[qid]["score"],
                "gpt35": review_gpt35[qid]["score"],
            },
        }

        # cleanup data
        cleaned_evals = {}
        for k, v in r["evaluations"].items():
            v = v.strip()
            lines = v.split("\n")
            # trim the first line if it's a pair of numbers
            if re.match(r"\d+[, ]+\d+", lines[0]):
                lines = lines[1:]
            v = "\n".join(lines)
            cleaned_evals[k] = v.replace("Assistant 1", "**Assistant 1**").replace(
                "Assistant 2", "**Assistant 2**"
            )

        r["evaluations"] = cleaned_evals
        records.append(r)

    # Reorder the records, this is optional
    for r in records:
        if r["id"] <= 20:
            r["id"] += 60
        else:
            r["id"] -= 20
    for r in records:
        if r["id"] <= 50:
            r["id"] += 10
        elif 50 < r["id"] <= 60:
            r["id"] -= 50
    for r in records:
        if r["id"] == 7:
            r["id"] = 1
        elif r["id"] < 7:
            r["id"] += 1

    records.sort(key=lambda x: x["id"])

    # Write to file
    with open("webpage/data.json", "w") as f:
        json.dump({"questions": records, "models": models}, f, indent=2)
