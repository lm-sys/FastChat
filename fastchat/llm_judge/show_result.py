"""
Usage:
python3 show_result.py --mode [single|pairwise-baseline|pairwise-all]
"""
import argparse
import glob
import os
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import mlflow

CATEGORIES = ["Writing", "Roleplay", "Reasoning", "Math", "Coding", "Extraction", "STEM", "Humanities"]

def get_model_df(input_file):
    cnt = 0
    q2result = []
    fin = open(input_file, "r")
    for line in fin:
        obj = json.loads(line)
        obj["category"] = CATEGORIES[(obj["question_id"]-81)//10]
        q2result.append(obj)
    df = pd.DataFrame(q2result)
    return df

# https://colab.research.google.com/drive/15O3Y8Rxq37PuMlArE291P4OC6ia37PQK#scrollTo=5i8R0l-XqkgO
def plot_result_single(input_file):
    df = get_model_df(input_file)
    
    all_models = df["model"].unique()
    print(all_models)
    scores_all = []
    for model in all_models:
        for cat in CATEGORIES:
            # filter category/model, and score format error (<1% case)
            res = df[(df["category"]==cat) & (df["model"]==model) & (df["score"] >= 0)]
            score = res["score"].mean()

            # # pairwise result
            # res_pair = df_pair[(df_pair["category"]==cat) & (df_pair["model"]==model)]["result"].value_counts()
            # wincnt = res_pair["win"] if "win" in res_pair.index else 0
            # tiecnt = res_pair["tie"] if "tie" in res_pair.index else 0
            # winrate = wincnt/res_pair.sum()
            # winrate_adjusted = (wincnt + tiecnt)/res_pair.sum()
            # # print(winrate_adjusted)

            # scores_all.append({"model": model, "category": cat, "score": score, "winrate": winrate, "wtrate": winrate_adjusted})
            scores_all.append({"model": model, "category": cat, "score": score})

    df_score = pd.DataFrame(scores_all)
    
    fig = px.line_polar(df_score, r = 'score', theta = 'category', line_close = True, category_orders = {"category": CATEGORIES},
                    color = 'model', markers=True, color_discrete_sequence=px.colors.qualitative.Pastel)

    fig_filename = "mtb_radar.png"

    fig.write_image(fig_filename)
    mlflow.log_artifact(fig_filename)
    fig.show()


def display_result_single(args):
    if args.input_file is None:
        input_file = (
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_single.jsonl"
        )
    else:
        input_file = args.input_file

    print(f"Input file: {input_file}")
    df_all = pd.read_json(input_file, lines=True)
    df = df_all[["model", "score", "turn"]]
    df = df[df["score"] != -1]

    if args.model_list is not None:
        df = df[df["model"].isin(args.model_list)]

    print("\n########## Judgement Model ##########")
    print(os.environ.get("AZURE_OPENAI_ENGINE", "tscience-uks-gpt-4o"))
    print("### Note that different judgement models may lead to significantly different scores ###")

    print("\n########## First turn ##########")
    df_1 = df[df["turn"] == 1].groupby(["model", "turn"]).mean()
    print(df_1.sort_values(by="score", ascending=False))

    if args.bench_name == "mt_bench":
        print("\n########## Second turn ##########")
        df_2 = df[df["turn"] == 2].groupby(["model", "turn"]).mean()
        print(df_2.sort_values(by="score", ascending=False))

        print("\n########## Average ##########")
        df_3 = df[["model", "score"]].groupby(["model"]).mean()
        print(df_3.sort_values(by="score", ascending=False))

        print("\n########## Category Average ##########")
        df_category = df_all[["model", "score", "question_id", "turn"]]
        df_category = df_category[df_category["score"] != -1]
        if args.model_list is not None:
            df_category = df_category[df_category["model"].isin(args.model_list)]
        df_category["category"] = df_category["question_id"].apply(
            lambda x: CATEGORIES[(x-81)//10]
        )
        df_4 = df_category[["model", "category", "score"]].groupby(["model", "category"]).mean()
        print(df_4)

        plot_result_single(input_file)


def display_result_pairwise(args):
    if args.input_file is None:
        input_file = (
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_pair.jsonl"
        )
    else:
        input_file = args.input_file

    print(f"Input file: {input_file}")
    df_all = pd.read_json(input_file, lines=True)
    df_all = df_all[(df_all["g1_winner"] != "error") & (df_all["g2_winner"] != "error")]

    model_list = (
        df_all["model_1"].unique().tolist() + df_all["model_2"].unique().tolist()
    )
    model_list = list(set(model_list))

    list_res = []
    # traverse df row by row
    for index, row in df_all.iterrows():
        if args.model_list is not None and row["model_1"] not in args.model_list:
            continue
        if args.baseline_model is not None:
            if args.baseline_model not in [row["model_1"], row["model_2"]]:
                continue
        if row["g1_winner"] == "tie" or row["g1_winner"] != row["g2_winner"]:
            list_res.append({"model": row["model_1"], "win": 0, "loss": 0, "tie": 1})
            list_res.append({"model": row["model_2"], "win": 0, "loss": 0, "tie": 1})
        else:
            if row["g1_winner"] == "model_1":
                winner = row["model_1"]
                loser = row["model_2"]
            else:
                winner = row["model_2"]
                loser = row["model_1"]
            list_res.append({"model": winner, "win": 1, "loss": 0, "tie": 0})
            list_res.append({"model": loser, "win": 0, "loss": 1, "tie": 0})

    df = pd.DataFrame(list_res)
    df = df.groupby(["model"]).sum()

    # remove baseline model
    if args.baseline_model is not None:
        df = df[df.index != args.baseline_model]
    # add win rate
    df["win_rate"] = df["win"] / (df["win"] + df["loss"] + df["tie"])
    df["loss_rate"] = df["loss"] / (df["win"] + df["loss"] + df["tie"])
    # each tie counts as 0.5 win + 0.5 loss
    df["win_rate_adjusted"] = (df["win"] + 0.5 * df["tie"]) / (
        df["win"] + df["loss"] + df["tie"]
    )
    # print(df.sort_values(by="win_rate", ascending=False))
    # print(df.sort_values(by="loss_rate", ascending=True))
    print(df.sort_values(by="win_rate_adjusted", ascending=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--judge-model", type=str, default="gpt-4")
    parser.add_argument("--baseline-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["pairwise-baseline", "pairwise-all", "single"],
        help=(
            "Evaluation mode. "
            "`pairwise-baseline` runs pairwise comparision against a baseline. "
            "`pairwise-all` runs pairwise comparision between all pairs. "
            "`single` runs single answer grading."
        ),
    )
    parser.add_argument(
        "--judgment_dir",
        type=str,
        default=None,
        help="dir where judgment file is picked up from",
    )
    args = parser.parse_args()

    if args.mode == "single":
        display_result_func = display_result_single
    else:
        if args.mode == "pairwise-all":
            args.baseline_model = None
        display_result_func = display_result_pairwise

    print(f"Mode: {args.mode}")

    if args.judgment_dir:
        filenames = glob.glob(os.path.join(args.judgment_dir, "*.jsonl"))
        assert len(filenames) == 1, "support 1 judgment file only"
        args.input_file = filenames[0]

    display_result_func(args)