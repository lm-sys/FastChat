"""A gradio app that renders a static leaderboard. This is used for Hugging Face Space."""
import argparse
import pickle

import gradio as gr


notebook_url = "https://colab.research.google.com/drive/17L9uCiAivzWfzOxo2Tb9RMauT7vS6nVU?usp=sharing"


def make_leaderboard_md(elo_results):
    leaderboard_md = f"""
# Leaderboard
| [Vote](https://arena.lmsys.org/) | [Blog](https://lmsys.org/blog/2023-05-03-arena/) | [GitHub](https://github.com/lm-sys/FastChat) | [Paper](https://arxiv.org/abs/2306.05685) | [Twitter](https://twitter.com/lmsysorg) | [Discord](https://discord.gg/HSWAKCrnFx) |

We use the Elo rating system to calculate the relative performance of the models. You can view the voting data, basic analyses, and calculation procedure in this [notebook]({notebook_url}). We will periodically release new leaderboards. If you want to see more models, please help us [add them](https://github.com/lm-sys/FastChat/blob/main/docs/arena.md#how-to-add-a-new-model).
Last updated: {elo_results["last_updated_datetime"]}
{elo_results["leaderboard_table"]}
"""
    return leaderboard_md


def build_leaderboard_tab(elo_results_file):
    if elo_results_file is not None:
        with open(elo_results_file, "rb") as fin:
            elo_results = pickle.load(fin)

        md = make_leaderboard_md(elo_results)
        p1 = elo_results["win_fraction_heatmap"]
        p2 = elo_results["battle_count_heatmap"]
        p3 = elo_results["average_win_rate_bar"]
        p4 = elo_results["bootstrap_elo_rating"]
    else:
        md = "Loading ..."
        p1 = p2 = p3 = p4 = None

    md_1 = gr.Markdown(md)
    gr.Markdown(
        f"""## More Statistics\n
We added some additional figures to show more statistics. The code for generating them is also included in this [notebook]({notebook_url}).
Please note that you may see different orders from different ranking methods. This is expected for models that perform similarly, as demonstrated by the confidence interval in the bootstrap figure. Going forward, we prefer the classical Elo calculation because of its scalability and interpretability. You can find more discussions in this blog [post](https://lmsys.org/blog/2023-05-03-arena/).
"""
    )

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
                "#### Figure 3: Average Win Rate Against All Other Models (Assuming Uniform Sampling and No Ties)"
            )
            plot_3 = gr.Plot(p3, show_label=False)
        with gr.Column():
            gr.Markdown(
                "#### Figure 4: Bootstrap of Elo Estimates (1000 Rounds of Random Sampling)"
            )
            plot_4 = gr.Plot(p4, show_label=False)
    return [md_1, plot_1, plot_2, plot_3, plot_4]


def build_demo(elo_results_file):
    with gr.Blocks(
        title="Chatbot Arena Leaderboard",
        theme=gr.themes.Base(),
    ) as demo:
        leader_components = build_leaderboard_tab(elo_results_file)

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_demo("elo_results_20230619.pkl")
    demo.launch(share=args.share)
