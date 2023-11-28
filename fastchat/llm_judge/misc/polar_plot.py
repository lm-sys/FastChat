import json
import argparse

import pandas as pd
import plotly.express as px
from pathlib import Path

CATEGORIES = [
    "Writing",
    "Roleplay",
    "Reasoning",
    "Math",
    "Coding",
    "Extraction",
    "STEM",
    "Humanities",
]


def get_model_df(model_judgment_fn):
    q2result = []
    fin = open(model_judgment_fn, "r")
    for line in fin:
        obj = json.loads(line)
        obj["category"] = CATEGORIES[(obj["question_id"] - 81) // 10]
        q2result.append(obj)
    df = pd.DataFrame(q2result)
    return df


def make_polar_plot(cfg_fn: str):
    # Load configuration
    cfg = json.load(open(cfg_fn, "r"))

    # Output directory
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open the jsonl file
    model_judgment_fn = cfg["model_judgment_fn"]
    df = get_model_df(model_judgment_fn)
    all_models = df["model"].unique()
    print(f"All models available in {model_judgment_fn}: {all_models}")
    scores_all = []
    for model in all_models:
        for cat in CATEGORIES:
            # filter category/model, and score format error (<1% case)
            res = df[
                (df["category"] == cat) & (df["model"] == model) & (df["score"] >= 0)
            ]
            score = res["score"].mean()

            scores_all.append({"model": model, "category": cat, "score": score})

    # Choose subset of models
    target_models = cfg["target_models"]

    scores_target = [
        scores_all[i]
        for i in range(len(scores_all))
        if scores_all[i]["model"] in target_models
    ]

    # sort by target_models
    scores_target = sorted(
        scores_target, key=lambda x: target_models.index(x["model"]), reverse=True
    )

    df_score = pd.DataFrame(scores_target)
    df_score = df_score[df_score["model"].isin(target_models)]
    df_score.to_excel(output_dir / "models_score.xlsx")
    for model in target_models:
        mean_score = df_score.query(f"model=='{model}'")["score"].mean()
        print(f"{model}: {mean_score:.3f}")

    rename_map = cfg["rename_map"]

    for k, v in rename_map.items():
        df_score.replace(k, v, inplace=True)

    fig = px.line_polar(
        df_score,
        r="score",
        theta="category",
        line_close=True,
        category_orders={"category": CATEGORIES},
        color="model",
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Pastel,
        range_r=[0, 10],
    )

    fig.update_layout(
        font=dict(
            size=18,
        ),
    )
    fig.write_image(output_dir / "polar_plot.png", width=800, height=600, scale=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_fn", type=str, required=True)
    args = parser.parse_args()

    print(f"Make polar plot with {args.cfg_fn}...")
    make_polar_plot(args.cfg_fn)
