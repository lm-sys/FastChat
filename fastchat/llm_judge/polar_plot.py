import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

CATEGORIES = ["Writing", "Roleplay", "Reasoning", "Math", "Coding", "Extraction", "STEM", "Humanities"]


def get_model_df():
    cnt = 0
    q2result = []
    fin = open("data/mt_bench/model_judgment/gpt-4_single-download.jsonl", "r")
    for line in fin:
        obj = json.loads(line)
        obj["category"] = CATEGORIES[(obj["question_id"]-81)//10]
        q2result.append(obj)
    df = pd.DataFrame(q2result)
    return df

def toggle(res_str):
    if res_str == "win":
        return "loss"
    elif res_str == "loss":
        return "win"
    return "tie"

if __name__ == "__main__":
    # Output directory
    output_dir = Path("outputs") / "single_2"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open the jsonl file
    df = get_model_df()
    all_models = df["model"].unique()
    print(all_models)
    scores_all = []
    for model in all_models:
        for cat in CATEGORIES:
            # filter category/model, and score format error (<1% case)
            res = df[(df["category"]==cat) & (df["model"]==model) & (df["score"] >= 0)]
            score = res["score"].mean()

            # scores_all.append({"model": model, "category": cat, "score": score, "winrate": winrate, "wtrate": winrate_adjusted})
            scores_all.append({"model": model, "category": cat, "score": score})

    # Choose subset of models
    target_models = [
        # "llama-13b", "alpaca-13b", "vicuna-13b-v1.3", "vicuna-33b-v1.3",
        'Llama-2-70b-chat', 'palm-2-chat-bison-001', "gpt-3.5-turbo",
        "claude-v1", "gpt-4", "gpt-4-1106-preview"
    ]

    scores_target = [scores_all[i] for i in range(len(scores_all)) if scores_all[i]["model"] in target_models]

    # sort by target_models
    scores_target = sorted(scores_target, key=lambda x: target_models.index(x["model"]), reverse=True)

    df_score = pd.DataFrame(scores_target)
    df_score = df_score[df_score["model"].isin(target_models)]
    df_score.to_excel(output_dir / "models_score.xlsx")
    for model in target_models:
        mean_score = df_score.query(f"model=='{model}'")["score"].mean()
        print(f"{model}: {mean_score:.3f}")

    rename_map = {
        # "llama-13b": "LLaMA-13B",
        # "alpaca-13b": "Alpaca-13B",
        # "vicuna-33b-v1.3": "Vicuna-33B",
        # "vicuna-13b-v1.3": "Vicuna-13B",
        "Llama-2-70b-chat": "LLaMA-2-70B Chat",
        "palm-2-chat-bison-001": "PaLM-2 Chat Bison",
        "gpt-3.5-turbo": "GPT-3.5-turbo",
        "claude-v1": "Claude-v1",
        "gpt-4": "GPT-4",
        "gpt-4-1106-preview": "GPT-4-turbo",
    }

    for k, v in rename_map.items():
        df_score.replace(k, v, inplace=True)

    fig = px.line_polar(df_score, r = 'score', theta = 'category', line_close = True, category_orders = {"category": CATEGORIES},
                        color = 'model', markers=True, color_discrete_sequence=px.colors.qualitative.Pastel)

    fig.update_layout(
        font=dict(
            size=18,
        ),
    )
    fig.write_image(output_dir / "polar_plot.png", width=800, height=600, scale=2)