import pandas as pd
import pickle
import gradio as gr

from fastchat.constants import SURVEY_LINK

deprecated_model_name = [
    "gemini-1.5-pro-exp-0801",
    "gemini-1.5-pro-api-0409-preview",
    "bard-jan-24-gemini-pro",
    "chatgpt-4o-latest-20240808",
    "gemini-1.5-pro-exp-0827",
    "gemini-1.5-flash-exp-0827",
    "chatgpt-4o-latest-20240903",
    "gemini-test-13",
    "gemini-test-11-si-v2",
    "gemini-test-11-v2",
    "gemini-test-9",
    "gemini-1.5-pro-test-6",
    "gemini-1.5-pro-test-7-si",
    "gemini-1.5-pro-test-7",
    "gemini-exp-1114",
    "gemini-exp-1121",
    "gemini-1.5-flash-8b-exp-0827",
    "gemini-2.0-flash-thinking-exp-1219",
    "yi-lightning-lite",
    "yi-large",
    "yi-large-preview",
    "qwen-plus-0828",
    "qwen-max-0428",
    "claude-1",
    "claude-2.0",
    "claude-2.1",
    "claude-instant-1",
    "deepseek-coder-v2-0724",
    "deepseek-v2-api-0628",
    "mistral-next",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-0314",
    "pplx-7b-online",
    "pplx-70b-online",
    "reka-core-20240501",
    "reka-core-20240722",
    "reka-flash-preview-20240611",
    "reka-flash-20240722",
    "reka-flash-21b",
    "reka-flash-21b-online",
    "glm-4-0116",
    "gemini-exp-1206",
    "gemini-2.0-flash-exp",
    "chatgpt-4o-latest-20241120",
    "early-grok-3",
    "chatgpt-4o-latest-20250129",
    "llama-4-maverick-03-26-experimental",
]

key_to_category_name = {
    "full": "Overall",
    "dedup": "De-duplicate Top Redundant Queries (soon to be default)",
    "math": "Math",
    "if": "Instruction Following",
    "multiturn": "Multi-Turn",
    "creative_writing": "Creative Writing",
    "creative_writing_vision": "Creative Writing",
    "coding": "Coding",
    "hard_6": "Hard Prompts",
    "hard_english_6": "Hard Prompts (English)",
    "long_user": "Longer Query",
    "english": "English",
    "chinese": "Chinese",
    "french": "French",
    "german": "German",
    "spanish": "Spanish",
    "russian": "Russian",
    "japanese": "Japanese",
    "korean": "Korean",
    "no_tie": "Exclude Ties",
    "no_short": "Exclude Short Query (< 5 tokens)",
    "no_refusal": "Exclude Refusal",
    "overall_limit_5_user_vote": "overall_limit_5_user_vote",
    "full_old": "Overall (Deprecated)",
    "captioning": "Captioning",
    "entity_recognition": "Entity Recognition",
    "ocr": "OCR",
    "humor": "Humor",
    "homework": "Homework",
    "diagram": "Diagram",
    "is_preset": "Exclude Preset Images",
}
cat_name_to_explanation = {
    "Overall": "Overall Questions",
    "De-duplicate Top Redundant Queries (soon to be default)": "De-duplicate top redundant queries (top 0.1%). See details in [blog post](https://lmsys.org/blog/2024-05-17-category-hard/#note-enhancing-quality-through-de-duplication).",
    "Math": "Math",
    "Instruction Following": "Instruction Following",
    "Multi-Turn": "Multi-Turn Conversation (>= 2 turns)",
    "Coding": "Coding: whether conversation contains code snippets",
    "Hard Prompts": "Hard Prompts: details in [blog post](https://lmsys.org/blog/2024-05-17-category-hard/)",
    "Hard Prompts (English)": "Hard Prompts (English), note: the delta is to English Category. details in [blog post](https://lmsys.org/blog/2024-05-17-category-hard/)",
    "Longer Query": "Longer Query (>= 500 tokens)",
    "English": "English Prompts",
    "Chinese": "Chinese Prompts",
    "French": "French Prompts",
    "German": "German Prompts",
    "Spanish": "Spanish Prompts",
    "Russian": "Russian Prompts",
    "Japanese": "Japanese Prompts",
    "Korean": "Korean Prompts",
    "Exclude Ties": "Exclude Ties and Bothbad",
    "Exclude Short Query (< 5 tokens)": "Exclude Short User Query (< 5 tokens)",
    "Exclude Refusal": 'Exclude model responses with refusal (e.g., "I cannot answer")',
    "overall_limit_5_user_vote": "overall_limit_5_user_vote",
    "Overall (Deprecated)": "Overall without De-duplicating Top Redundant Queries (top 0.1%). See details in [blog post](https://lmsys.org/blog/2024-05-17-category-hard/#note-enhancing-quality-through-de-duplication).",
    "Creative Writing": "Creative Writing",
    "Exclude Preset Images": "Exclude Images from 'Random Example' Option",
    "Captioning": "Open-Ended Captioning",
    "Entity Recognition": "Entity Recognition (e.g. who is in the image)",
    "OCR": "Optical Character Recognition",
    "Humor": "Humor (e.g. writing jokes, meme understanding)",
    "Homework": "Homework problems",
    "Diagram": "Diagram (e.g. plots, flow charts, figures)",
}
cat_name_to_baseline = {
    "Hard Prompts (English)": "English",
}

notebook_url = (
    "https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH"
)

basic_component_values = [None] * 6
leader_component_values = [None] * 5


def make_default_md_1(mirror=False):
    link_color = "#1976D2"  # This color should be clear in both light and dark mode
    leaderboard_md = f"""
    # 🏆 Chatbot Arena LLM Leaderboard: Community-driven Evaluation for Best LLM and AI chatbots
    [Twitter](https://twitter.com/lmarena_ai) | [Discord](https://discord.gg/6GXcFg3TH8) | [Blog](https://blog.lmarena.ai/) | [GitHub](https://github.com/lm-sys/FastChat) | [Paper](https://arxiv.org/abs/2403.04132) | [Dataset](https://github.com/lm-sys/FastChat/blob/main/docs/dataset_release.md) | [Kaggle Competition](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena)
    """

    return leaderboard_md


def make_default_md_2(mirror=False):
    mirror_str = "<span style='color: red; font-weight: bold'>This is a mirror of the live leaderboard created and maintained at <a href='https://lmarena.ai/leaderboard' style='color: #B00020; text-decoration: none;'>https://lmarena.ai/leaderboard</a>. Please link to the original URL for citation purposes.</span>"
    leaderboard_md = f"""
{mirror_str if mirror else ""}

Chatbot Arena ([lmarena.ai](https://lmarena.ai)) is an open-source platform for evaluating AI through human preference, developed by researchers at [LMArena](https://blog.lmarena.ai/about/). With over 1,000,000 user votes, the platform ranks best LLM and AI chatbots using the Bradley-Terry model to generate live leaderboards. For technical details, check out our [paper](https://arxiv.org/abs/2403.04132).

**Chatbot Arena thrives on community engagement — cast your vote to help improve AI evaluation!**

{SURVEY_LINK}
"""

    return leaderboard_md


def make_arena_leaderboard_md(arena_df, last_updated_time, vision=False):
    total_votes = sum(arena_df["num_battles"]) // 2
    total_models = len(arena_df)
    space = "&nbsp;&nbsp;&nbsp;"

    leaderboard_md = f"""
Total #models: **{total_models}**.{space} Total #votes: **{"{:,}".format(total_votes)}**.{space} Last updated: {last_updated_time}.
"""

    leaderboard_md += f"""
Code to recreate leaderboard tables and plots in this [notebook]({notebook_url}). You can contribute your vote at [lmarena.ai](https://lmarena.ai)!
"""
    return leaderboard_md


def make_category_arena_leaderboard_md(arena_df, arena_subset_df, name="Overall"):
    total_votes = sum(arena_df["num_battles"]) // 2
    total_models = len(arena_df)
    space = "&nbsp;&nbsp;&nbsp;"
    total_subset_votes = sum(arena_subset_df["num_battles"]) // 2
    total_subset_models = len(arena_subset_df)
    if "w/ Style Control" in name:
        explanation = (
            cat_name_to_explanation[name.replace(" w/ Style Control", "")]
            + " with Style Control. See details in [blog post](https://lmsys.org/blog/2024-08-28-style-control/)."
        )
    else:
        explanation = cat_name_to_explanation[name]
    leaderboard_md = f"""### {explanation}
#### {space} #models: **{total_subset_models} ({round(total_subset_models/total_models *100)}%)** {space} #votes: **{"{:,}".format(total_subset_votes)} ({round(total_subset_votes/total_votes * 100)}%)**{space}
"""
    return leaderboard_md


def make_full_leaderboard_md():
    leaderboard_md = """
Three benchmarks are displayed: **Arena Elo**, **MT-Bench** and **MMLU**.
- [Chatbot Arena](https://lmarena.ai) - a crowdsourced, randomized battle platform. We use 1M+ user votes to compute model strength.
- [MT-Bench](https://arxiv.org/abs/2306.05685): a set of challenging multi-turn questions. We use GPT-4 to grade the model responses.
- [MMLU](https://arxiv.org/abs/2009.03300) (5-shot): a test to measure a model's multitask accuracy on 57 tasks.

💻 Code: The MT-bench scores (single-answer grading on a scale of 10) are computed by [fastchat.llm_judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge).
The MMLU scores are mostly computed by [InstructEval](https://github.com/declare-lab/instruct-eval).
Higher values are better for all benchmarks. Empty cells mean not available.
"""
    return leaderboard_md


def make_leaderboard_md_live(elo_results):
    leaderboard_md = f"""
# Leaderboard
Last updated: {elo_results["last_updated_datetime"]}
{elo_results["leaderboard_table"]}
"""
    return leaderboard_md


def arena_hard_title(date):
    arena_hard_title = f"""
Last Updated: {date}

**Arena-Hard-Auto v0.1** - an automatic evaluation tool for instruction-tuned LLMs with 500 challenging user queries curated from Chatbot Arena. 

We prompt GPT-4-Turbo as judge to compare the models' responses against a baseline model (default: GPT-4-0314). If you are curious to see how well your model might perform on Chatbot Arena, we recommend trying Arena-Hard-Auto. Check out our paper for more details about how Arena-Hard-Auto works as an fully automated data pipeline converting crowdsourced data into high-quality benchmarks ->
[[Paper](https://arxiv.org/abs/2406.11939) | [Repo](https://github.com/lm-sys/arena-hard-auto)]
    """
    return arena_hard_title
