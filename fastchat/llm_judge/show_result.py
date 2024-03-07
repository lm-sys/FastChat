"""
Usage:
python3 show_result.py --mode [single|pairwise-baseline|pairwise-all]
"""
import argparse
import pandas as pd

pd.set_option('display.max_rows', None)


def display_result_single(args):
    # Adjust display settings to prevent cutoff
    pd.set_option('display.max_colwidth', None)  # Avoid column value cutoff
    pd.set_option('display.width', None)  # Use maximum width available

    if args.input_file is None:
        input_file = f"data/{args.bench_name}/model_judgment/{args.judge_model}_single.jsonl"
    else:
        input_file = args.input_file

    print(f"Input file: {input_file}")
    df_all = pd.read_json(input_file, lines=True)
    print(df_all.columns)
    # Index(['question_id', 'model', 'judge', 'user_prompt', 'judgment', 'score',
    #   'turn', 'tstamp'],
    # Remove any duplicate (question_id, model, user_prompt, turn) tuples
    df_all = df_all.drop_duplicates(subset=["question_id", "model", "user_prompt", "turn"])
    df = df_all[["model", "score", "turn", "question_id"]]
    df = df[df["score"] != -1]

    if args.model_list is not None:
        df = df[df["model"].isin(args.model_list)]
    # loop through models and show how many outputs are stored
    for model in df["model"].unique():
        print(f"Model: {model}, number of outputs: {len(df[df['model'] == model])}")
    
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

    show_relative_drop(df)
    df_turn_1 = df[df["turn"] == 1]
    show_relative_drop(df_turn_1)


def show_relative_drop(df):
        
    suffixes_1 = [
        # "",
        # "_coeff_1.5_refusal_data_full_answers",
        "_coeff_-1.5_refusal_data_full_answers",
        # "_coeff_1.5_refusal_data_A_B_question_pairs",
        "_coeff_-1.5_refusal_data_A_B_question_pairs",
    ]
    suffixes_2 = [
        "_coeff_1.5_refusal_data_full_answers",
        # "_coeff_-1.5_refusal_data_full_answers",
        "_coeff_1.5_refusal_data_A_B_question_pairs",
        # "_coeff_-1.5_refusal_data_A_B_question_pairs",
    ]

    suffixes = suffixes_1 + suffixes_2
    # For all base models, show the average drop in score when the suffix is added
    base_models = [model for model in df["model"].unique() if not any(suffix in model for suffix in suffixes)]
    for base_model in base_models:
        # Check if all suffixes are contained for this base model
        if (not all([base_model + suffix in df["model"].unique() for suffix in suffixes_1]) and
                not all([base_model + suffix in df["model"].unique() for suffix in suffixes_2])):
            continue
        print(f"\n########## {base_model} ##########")
        all_diffs = []
        min_diff = -1000
        df_base = df[df["model"] == base_model]
        existing_suffixes = [suffix for suffix in suffixes if base_model + suffix in df["model"].unique()]
        for suffix in existing_suffixes:
            model = base_model + suffix
            df_model = df[df["model"] == model]
            df_diff = df_model.merge(df_base, on=["turn", "question_id"], suffixes=("_" + suffix, "_base"))
            df_diff["diff"] = df_diff["score_" + suffix] - df_diff["score_base"]
            diff = df_diff['diff'].mean()
            all_diffs.append(diff)
            min_diff = max(min_diff, diff)
            print(f"Suffix: {suffix}, diff: {diff}")
        original_score = df_base['score'].mean()
        average_diff = sum(all_diffs) / len(all_diffs)
        average_diff_fraction = average_diff / original_score
        print(f"Original score: {original_score} Min diff: {min_diff}, average diff: {average_diff}, average diff fraction: {average_diff_fraction}")
    
    

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
    args = parser.parse_args()

    if args.mode == "single":
        display_result_func = display_result_single
    else:
        if args.mode == "pairwise-all":
            args.baseline_model = None
        display_result_func = display_result_pairwise

    print(f"Mode: {args.mode}")
    display_result_func(args)
