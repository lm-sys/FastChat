"""
Usage:
python3 show_result.py --mode [single|pairwise-baseline|pairwise-all]
"""
import argparse
import pandas as pd


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

    df_1 = df[df["turn"] == 1].groupby(["model", "turn"]).mean()
    print(df_1.sort_values(by="score", ascending=False))

    if args.bench_name == "mt_bench":
        df_2 = df[df["turn"] == 2].groupby(["model", "turn"]).mean()
        print(df_2.sort_values(by="score", ascending=False))

        df_3 = df[["model", "score"]].groupby(["model"]).mean()
        print(df_3.sort_values(by="score", ascending=False))


def display_result_pairwise(args):
    if args.input_file is None:
        input_file = (
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_pair.jsonl"
        )
    else:
        input_file = args.input_file

    print(f"Input file: {input_file}")
    df_all = pd.read_json(input_file, lines=True)
    model_list = (
        df_all["model_1"].unique().tolist() + df_all["model_2"].unique().tolist()
    )
    model_list = list(set(model_list))

    list_res = []
    # traverse df row by row
    for index, row in df_all.iterrows():
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
    print(df.sort_values(by="win_rate", ascending=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--judge-model", type=str, default="gpt-4")
    parser.add_argument("--baseline-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--mode",
        type=str,
        default="pairwise-baseline",
        choices=["pairwise-baseline", "pairwise-all", "single"],
        help=(
            "Evaluation mode. "
            "`pairwise-baseline` runs pairwise comparision against a baseline. "
            "`pairwise-all` runs pairwise comparision between all pairs. "
            "`single` runs single answer grading."
        ),
    )
    args = parser.parse_args()

    if args.mode == "single":
        display_result_func = display_result_single
    else:
        if args.mode == "pairwise-all":
            args.baseline_model = None
        display_result_func = display_result_pairwise

    print(f"Mode: {args.mode}")
    display_result_func(args)
