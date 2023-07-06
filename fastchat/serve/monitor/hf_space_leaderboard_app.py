"""A gradio app that renders a static leaderboard. This is used for Hugging Face Space."""
import ast
import argparse
import pickle

import gradio as gr
import numpy as np


notebook_url = "https://colab.research.google.com/drive/1RAWb22-PFNI-X1gPVzc927SGUdfr6nsR?usp=sharing"


basic_component_values = [None] * 6
leader_component_values = [None] * 5


def make_leaderboard_md(elo_results):
    leaderboard_md = f"""
# Leaderboard
| [Blog](https://lmsys.org/blog/2023-05-03-arena/) | [GitHub](https://github.com/lm-sys/FastChat) | [Paper](https://arxiv.org/abs/2306.05685) | [Twitter](https://twitter.com/lmsysorg) | [Discord](https://discord.gg/HSWAKCrnFx) |

🏆 This leaderboard is based on the following three benchmarks.
- [Chatbot Arena](https://lmsys.org/blog/2023-05-03-arena/) - a crowdsourced, randomized battle platform. We use 40K+ user votes to compute Elo ratings.
- [MT-Bench](https://arxiv.org/abs/2306.05685) - a set of challenging multi-turn questions. We use GPT-4 to grade the model responses.
- [MMLU](https://arxiv.org/abs/2009.03300) (5-shot) - a test to measure a model's multitask accuracy on 57 tasks.

💻 We use [fastchat.llm_judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) to compute MT-bench scores (single-answer grading on a scale of 10) and win rates (against gpt-3.5). The Arena Elo ratings are computed by this [notebook]({notebook_url}). The MMLU scores are computed by [InstructEval](https://github.com/declare-lab/instruct-eval) and [Chain-of-Thought Hub](https://github.com/FranxYao/chain-of-thought-hub). Higher values are better for all benchmarks. Empty cells mean not available.
"""
    return leaderboard_md


def make_leaderboard_md_live(elo_results):
    leaderboard_md = f"""
# Leaderboard
Last updated: {elo_results["last_updated_datetime"]}
{elo_results["leaderboard_table"]}
"""
    return leaderboard_md


def update_elo_components(max_num_files, elo_results_file):
    log_files = get_log_files(max_num_files)

    # Leaderboard
    if elo_results_file is None:  # Do live update
        battles = clean_battle_data(log_files)
        elo_results = report_elo_analysis_results(battles)

        leader_component_values[0] = make_leaderboard_md_live(elo_results)
        leader_component_values[1] = elo_results["win_fraction_heatmap"]
        leader_component_values[2] = elo_results["battle_count_heatmap"]
        leader_component_values[3] = elo_results["bootstrap_elo_rating"]
        leader_component_values[4] = elo_results["average_win_rate_bar"]

    # Basic stats
    basic_stats = report_basic_stats(log_files)
    md0 = f"Last updated: {basic_stats['last_updated_datetime']}"

    md1 = "### Action Histogram\n"
    md1 += basic_stats["action_hist_md"] + "\n"

    md2 = "### Anony. Vote Histogram\n"
    md2 += basic_stats["anony_vote_hist_md"] + "\n"

    md3 = "### Model Call Histogram\n"
    md3 += basic_stats["model_hist_md"] + "\n"

    md4 = "### Model Call (Last 24 Hours)\n"
    md4 += basic_stats["num_chats_last_24_hours"] + "\n"

    basic_component_values[0] = md0
    basic_component_values[1] = basic_stats["chat_dates_bar"]
    basic_component_values[2] = md1
    basic_component_values[3] = md2
    basic_component_values[4] = md3
    basic_component_values[5] = md4


def update_worker(max_num_files, interval, elo_results_file):
    while True:
        tic = time.time()
        update_elo_components(max_num_files, elo_results_file)
        durtaion = time.time() - tic
        print(f"update duration: {durtaion:.2f} s")
        time.sleep(max(interval - durtaion, 0))


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")
    return basic_component_values + leader_component_values


def model_hyperlink(model_name, link):
    return f'<a target="_blank" href="{link}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{model_name}</a>'


def load_leaderboard_table_csv(filename, add_hyperlink=True):
    lines = open(filename).readlines()
    heads = [v.strip() for v in lines[0].split(",")]
    rows = []
    for i in range(1, len(lines)):
        row = [v.strip() for v in lines[i].split(",")]
        for j in range(len(heads)):
            item = {}
            for h, v in zip(heads, row):
                if h == "Arena Elo rating":
                    if v != "-":
                        v = int(ast.literal_eval(v))
                    else:
                        v = np.nan
                elif h == "MMLU":
                    if v != "-":
                        v = round(ast.literal_eval(v) * 100, 1)
                    else:
                        v = np.nan
                elif h == "MT-bench (win rate %)":
                    if v != "-":
                        v = round(ast.literal_eval(v[:-1]), 1)
                    else:
                        v = np.nan
                elif h == "MT-bench (score)":
                    if v != "-":
                        v = round(ast.literal_eval(v), 2)
                    else:
                        v = np.nan
                item[h] = v
            if add_hyperlink:
                item["Model"] = model_hyperlink(item["Model"], item["Link"])
        rows.append(item)

    return rows


def build_basic_stats_tab():
    empty = "Loading ..."
    basic_component_values[:] = [empty, None, empty, empty, empty, empty]

    md0 = gr.Markdown(empty)
    gr.Markdown("#### Figure 1: Number of model calls and votes")
    plot_1 = gr.Plot(show_label=False)
    with gr.Row():
        with gr.Column():
            md1 = gr.Markdown(empty)
        with gr.Column():
            md2 = gr.Markdown(empty)
    with gr.Row():
        with gr.Column():
            md3 = gr.Markdown(empty)
        with gr.Column():
            md4 = gr.Markdown(empty)
    return [md0, plot_1, md1, md2, md3, md4]


def build_leaderboard_tab(elo_results_file, leaderboard_table_file):
    if elo_results_file is None:  # Do live update
        md = "Loading ..."
        p1 = p2 = p3 = p4 = None
    else:
        with open(elo_results_file, "rb") as fin:
            elo_results = pickle.load(fin)

        md = make_leaderboard_md(elo_results)
        p1 = elo_results["win_fraction_heatmap"]
        p2 = elo_results["battle_count_heatmap"]
        p3 = elo_results["bootstrap_elo_rating"]
        p4 = elo_results["average_win_rate_bar"]

    md_1 = gr.Markdown(md, elem_id="leaderboard_markdown")

    if leaderboard_table_file:
        data = load_leaderboard_table_csv(leaderboard_table_file)
        headers = [
            "Model",
            "Arena Elo rating",
            "MT-bench (score)",
            "MT-bench (win rate %)",
            "MMLU",
            "License",
        ]
        values = []
        for item in data:
            row = []
            for key in headers:
                value = item[key]
                row.append(value)
            values.append(row)
        values.sort(key=lambda x: -x[1] if not np.isnan(x[1]) else 1e9)

        headers[1] = "⭐ " + headers[1]
        headers[2] = "📈 " + headers[2]

        gr.Dataframe(
            headers=headers,
            datatype=["markdown", "number", "number", "number", "number", "str"],
            value=values,
            elem_id="leaderboard_dataframe",
        )
        gr.Markdown(
            "If you want to see more models, please help us [add them](https://github.com/lm-sys/FastChat/blob/main/docs/arena.md#how-to-add-a-new-model)."
        )
    else:
        pass

    gr.Markdown(
        f"""## More Statistics for Chatbot Arena\n
We added some additional figures to show more statistics. The code for generating them is also included in this [notebook]({notebook_url}).
Please note that you may see different orders from different ranking methods. This is expected for models that perform similarly, as demonstrated by the confidence interval in the bootstrap figure. Going forward, we prefer the classical Elo calculation because of its scalability and interpretability. You can find more discussions in this blog [post](https://lmsys.org/blog/2023-05-03-arena/).
"""
    )

    leader_component_values[:] = [md, p1, p2, p3, p4]

    with gr.Row():
        with gr.Column():
            gr.Markdown(
                "#### Figure 1: Fraction of Model A Wins for All Non-tied A vs. B Battles"
            )
            plot_1 = gr.Plot(p1, show_label=False)
        with gr.Column():
            gr.Markdown(
                "#### Figure 2: Battle Count for Each Combination of Models (without Ties)"
            )
            plot_2 = gr.Plot(p2, show_label=False)
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                "#### Figure 3: Bootstrap of Elo Estimates (1000 Rounds of Random Sampling)"
            )
            plot_3 = gr.Plot(p3, show_label=False)
        with gr.Column():
            gr.Markdown(
                "#### Figure 4: Average Win Rate Against All Other Models (Assuming Uniform Sampling and No Ties)"
            )
            plot_4 = gr.Plot(p4, show_label=False)
    return [md_1, plot_1, plot_2, plot_3, plot_4]


def build_demo(elo_results_file, leaderboard_table_file):
    text_size = gr.themes.sizes.text_lg

    with gr.Blocks(
        title="Chatbot Arena Leaderboard",
        theme=gr.themes.Base(text_size=text_size),
    ) as demo:
        leader_components = build_leaderboard_tab(
            elo_results_file, leaderboard_table_file
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_demo("elo_results_20230619.pkl", "leaderboard_table_20230619.csv")
    demo.launch(share=args.share)
