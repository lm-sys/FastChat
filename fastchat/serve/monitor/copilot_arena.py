import gradio as gr
import pandas as pd
import requests
import os

from fastchat.serve.monitor.monitor import recompute_final_ranking

copilot_arena_leaderboard_url = os.getenv("COPILOT_ARENA_LEADERBOARD_URL")


def process_copilot_arena_leaderboard(leaderboard):
    leaderboard = leaderboard.copy().loc[leaderboard["visibility"] == "public"]
    leaderboard["score"] = leaderboard["score"].round().astype(int)
    leaderboard["rating_q975"] = leaderboard["upper"].round().astype(int)
    leaderboard["rating_q025"] = leaderboard["lower"].round().astype(int)

    leaderboard["upper_diff"] = leaderboard["rating_q975"] - leaderboard["score"]
    leaderboard["lower_diff"] = leaderboard["score"] - leaderboard["rating_q025"]

    leaderboard["confidence_interval"] = (
        "+"
        + leaderboard["upper_diff"].astype(str)
        + " / -"
        + leaderboard["lower_diff"].astype(str)
    )

    rankings_ub = recompute_final_ranking(leaderboard)
    leaderboard.insert(loc=0, column="Rank* (UB)", value=rankings_ub)

    leaderboard = leaderboard.sort_values(
        by=["Rank* (UB)", "score"], ascending=[True, False]
    )

    return leaderboard


def build_copilot_arena_tab():
    response = requests.get(copilot_arena_leaderboard_url)
    if response.status_code == 200:
        leaderboard = pd.DataFrame(response.json()["elo_data"])
        leaderboard = process_copilot_arena_leaderboard(leaderboard)
        leaderboard = leaderboard.rename(
            columns={
                "name": "Model",
                "confidence_interval": "Confidence Interval",
                "score": "Arena Score",
                "organization": "Organization",
                "votes": "Votes",
            }
        )

        column_order = [
            "Rank* (UB)",
            "Model",
            "Arena Score",
            "Confidence Interval",
            "Votes",
            "Organization",
        ]
        leaderboard = leaderboard[column_order]
        num_models = len(leaderboard)
        total_battles = int(leaderboard["Votes"].sum()) // 2
        md = f"""
        [Copilot Arena](https://blog.lmarena.ai/blog/2024/copilot-arena/) is a free AI coding assistant that provides paired responses from different state-of-the-art LLMs. This leaderboard contains the relative performance and ranking of {num_models} models over {total_battles} battles.
        """

        gr.Markdown(md, elem_id="leaderboard_markdown")
        gr.DataFrame(
            leaderboard,
            datatype=["str" for _ in leaderboard.columns],
            elem_id="arena_hard_leaderboard",
            height=600,
            wrap=True,
            interactive=False,
            column_widths=[70, 130, 60, 80, 50, 80],
        )

        gr.Markdown(
            """
    ***Rank (UB)**: model's ranking (upper-bound), defined by one + the number of models that are statistically better than the target model.
    Model A is statistically better than model B when A's lower-bound score is greater than B's upper-bound score (in 95% confidence interval). \n
    **Confidence Interval**: represents the range of uncertainty around the Arena Score. It's displayed as +X / -Y, where X is the difference between the upper bound and the score, and Y is the difference between the score and the lower bound.
    """,
            elem_id="leaderboard_markdown",
        )
    else:
        gr.Markdown("Error with fetching Copilot Arena data. Check back in later.")
