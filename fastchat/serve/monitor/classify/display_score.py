import pandas as pd
import argparse
import os
from glob import glob
from sklearn.metrics import recall_score, precision_score

tag_names = {
    "if_bench": ("if_v0.1", "if"),
    "math_bench": ("math_v0.1", "math"),
    "hard_bench": ("criteria_v0.1", "hard"),
    "creative_writing_bench": ("creative_writing_v0.1", "creative_writing"),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=str, default="if_bench")
    parser.add_argument("--display-conflict", action="store_true")
    args = parser.parse_args()
    assert args.bench in tag_names, "Not valid bench argument, add bench if needed."

    test = pd.read_json(os.path.join("label_bench", args.bench, "test.json"))

    for file in glob(os.path.join("label_bench", args.bench, "data", "*.json")):
        output = pd.read_json(file)

        tag_map = (
            output[["question_id", "category_tag"]]
            .set_index("question_id")
            .to_dict("index")
        )

        tag_1, tag_2 = tag_names[args.bench]
        test["pred"] = test.question_id.map(
            lambda id: tag_map[id]["category_tag"][tag_1][tag_2]
        )

        accuracy = (test.label == test.pred).mean()
        recall = recall_score(y_pred=test.pred, y_true=test.label)
        precision = precision_score(y_pred=test.pred, y_true=test.label)

        print(f"Model: {output.model[0]}")
        print(f"Accuracy: {round(accuracy, 3)}")
        print(f"Precision: {round(precision, 3)}")
        print(f"Recall: {round(recall, 3)}")

        if args.display_conflict:
            print()
            print("###### CONFLICT ######")
            print()
            conflict = test[test.label & ~test.pred]
            print("Ground Truth = True; Pred = False")
            prompts = (
                conflict.conversation_a.map(lambda x: x[0]["content"])
                .sample(n=5)
                .tolist()
            )
            for prompt in prompts:
                print("####################")
                print(prompt)
            print()
            print()

            conflict = test[~test.label & test.pred]
            print("Ground Truth = False; Pred = True")
            prompts = (
                conflict.conversation_a.map(lambda x: x[0]["content"])
                .sample(n=5)
                .tolist()
            )
            for prompt in prompts:
                print("####################")
                print(prompt)
            print()
            print()
        print()
