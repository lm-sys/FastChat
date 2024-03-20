"""
Usage:
python3 qa_browser.py --share
"""

import argparse
from collections import defaultdict
import re

import gradio as gr

from fastchat.llm_judge.common import (
    load_questions,
    load_model_answers,
    load_single_model_judgments,
    load_pairwise_model_judgments,
    resolve_single_judgment_dict,
    resolve_pairwise_judgment_dict,
    get_single_judge_explanation,
    get_pairwise_judge_explanation,
)


questions = []
model_answers = {}

model_judgments_normal_single = {}
model_judgments_math_single = {}

model_judgments_normal_pairwise = {}
model_judgments_math_pairwise = {}

question_selector_map = {}
category_selector_map = defaultdict(list)


def display_question(category_selector, request: gr.Request):
    choices = category_selector_map[category_selector]
    return gr.Dropdown.update(
        value=choices[0],
        choices=choices,
    )


def display_pairwise_answer(
    question_selector, model_selector1, model_selector2, request: gr.Request
):
    q = question_selector_map[question_selector]
    qid = q["question_id"]

    ans1 = model_answers[model_selector1][qid]
    ans2 = model_answers[model_selector2][qid]

    chat_mds = pairwise_to_gradio_chat_mds(q, ans1, ans2)
    gamekey = (qid, model_selector1, model_selector2)

    judgment_dict = resolve_pairwise_judgment_dict(
        q,
        model_judgments_normal_pairwise,
        model_judgments_math_pairwise,
        multi_turn=False,
    )

    explanation = (
        "##### Model Judgment (first turn)\n"
        + get_pairwise_judge_explanation(gamekey, judgment_dict)
    )

    judgment_dict_turn2 = resolve_pairwise_judgment_dict(
        q,
        model_judgments_normal_pairwise,
        model_judgments_math_pairwise,
        multi_turn=True,
    )

    explanation_turn2 = (
        "##### Model Judgment (second turn)\n"
        + get_pairwise_judge_explanation(gamekey, judgment_dict_turn2)
    )

    return chat_mds + [explanation] + [explanation_turn2]


def display_single_answer(question_selector, model_selector1, request: gr.Request):
    q = question_selector_map[question_selector]
    qid = q["question_id"]

    ans1 = model_answers[model_selector1][qid]

    chat_mds = single_to_gradio_chat_mds(q, ans1)
    gamekey = (qid, model_selector1)

    judgment_dict = resolve_single_judgment_dict(
        q, model_judgments_normal_single, model_judgments_math_single, multi_turn=False
    )

    explanation = "##### Model Judgment (first turn)\n" + get_single_judge_explanation(
        gamekey, judgment_dict
    )

    judgment_dict_turn2 = resolve_single_judgment_dict(
        q, model_judgments_normal_single, model_judgments_math_single, multi_turn=True
    )

    explanation_turn2 = (
        "##### Model Judgment (second turn)\n"
        + get_single_judge_explanation(gamekey, judgment_dict_turn2)
    )

    return chat_mds + [explanation] + [explanation_turn2]


newline_pattern1 = re.compile("\n\n(\d+\. )")
newline_pattern2 = re.compile("\n\n(- )")


def post_process_answer(x):
    """Fix Markdown rendering problems."""
    x = x.replace("\u2022", "- ")
    x = re.sub(newline_pattern1, "\n\g<1>", x)
    x = re.sub(newline_pattern2, "\n\g<1>", x)
    return x


def pairwise_to_gradio_chat_mds(question, ans_a, ans_b, turn=None):
    end = len(question["turns"]) if turn is None else turn + 1

    mds = ["", "", "", "", "", "", ""]
    for i in range(end):
        base = i * 3
        if i == 0:
            mds[base + 0] = "##### User\n" + question["turns"][i]
        else:
            mds[base + 0] = "##### User's follow-up question \n" + question["turns"][i]
        mds[base + 1] = "##### Assistant A\n" + post_process_answer(
            ans_a["choices"][0]["turns"][i].strip()
        )
        mds[base + 2] = "##### Assistant B\n" + post_process_answer(
            ans_b["choices"][0]["turns"][i].strip()
        )

    ref = question.get("reference", ["", ""])

    ref_md = ""
    if turn is None:
        if ref[0] != "" or ref[1] != "":
            mds[6] = f"##### Reference Solution\nQ1. {ref[0]}\nQ2. {ref[1]}"
    else:
        x = ref[turn] if turn < len(ref) else ""
        if x:
            mds[6] = f"##### Reference Solution\n{ref[turn]}"
        else:
            mds[6] = ""
    return mds


def single_to_gradio_chat_mds(question, ans, turn=None):
    end = len(question["turns"]) if turn is None else turn + 1

    mds = ["", "", "", "", ""]
    for i in range(end):
        base = i * 2
        if i == 0:
            mds[base + 0] = "##### User\n" + question["turns"][i]
        else:
            mds[base + 0] = "##### User's follow-up question \n" + question["turns"][i]
        mds[base + 1] = "##### Assistant A\n" + post_process_answer(
            ans["choices"][0]["turns"][i].strip()
        )

    ref = question.get("reference", ["", ""])

    ref_md = ""
    if turn is None:
        if ref[0] != "" or ref[1] != "":
            mds[4] = f"##### Reference Solution\nQ1. {ref[0]}\nQ2. {ref[1]}"
    else:
        x = ref[turn] if turn < len(ref) else ""
        if x:
            mds[4] = f"##### Reference Solution\n{ref[turn]}"
        else:
            mds[4] = ""
    return mds


def build_question_selector_map():
    global question_selector_map, category_selector_map

    # Build question selector map
    for q in questions:
        preview = f"{q['question_id']}: " + q["turns"][0][:128] + "..."
        question_selector_map[preview] = q
        category_selector_map[q["category"]].append(preview)


def build_pairwise_browser_tab():
    global question_selector_map, category_selector_map

    models = list(model_answers.keys())
    num_sides = 2
    num_turns = 2
    side_names = ["A", "B"]

    question_selector_choices = list(question_selector_map.keys())
    category_selector_choices = list(category_selector_map.keys())

    # Selectors
    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            category_selector = gr.Dropdown(
                choices=category_selector_choices, label="Category", container=False
            )
        with gr.Column(scale=100):
            question_selector = gr.Dropdown(
                choices=question_selector_choices, label="Question", container=False
            )

    model_selectors = [None] * num_sides
    with gr.Row():
        for i in range(num_sides):
            with gr.Column():
                if i == 0:
                    value = models[0]
                else:
                    value = "gpt-3.5-turbo"
                model_selectors[i] = gr.Dropdown(
                    choices=models,
                    value=value,
                    label=f"Model {side_names[i]}",
                    container=False,
                )

    # Conversation
    chat_mds = []
    for i in range(num_turns):
        chat_mds.append(gr.Markdown(elem_id=f"user_question_{i+1}"))
        with gr.Row():
            for j in range(num_sides):
                with gr.Column(scale=100):
                    chat_mds.append(gr.Markdown())

                if j == 0:
                    with gr.Column(scale=1, min_width=8):
                        gr.Markdown()
    reference = gr.Markdown(elem_id=f"reference")
    chat_mds.append(reference)

    model_explanation = gr.Markdown(elem_id="model_explanation")
    model_explanation2 = gr.Markdown(elem_id="model_explanation")

    # Callbacks
    category_selector.change(display_question, [category_selector], [question_selector])
    question_selector.change(
        display_pairwise_answer,
        [question_selector] + model_selectors,
        chat_mds + [model_explanation] + [model_explanation2],
    )

    for i in range(num_sides):
        model_selectors[i].change(
            display_pairwise_answer,
            [question_selector] + model_selectors,
            chat_mds + [model_explanation] + [model_explanation2],
        )

    return (category_selector,)


def build_single_answer_browser_tab():
    global question_selector_map, category_selector_map

    models = list(model_answers.keys())
    num_sides = 1
    num_turns = 2
    side_names = ["A"]

    question_selector_choices = list(question_selector_map.keys())
    category_selector_choices = list(category_selector_map.keys())

    # Selectors
    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            category_selector = gr.Dropdown(
                choices=category_selector_choices, label="Category", container=False
            )
        with gr.Column(scale=100):
            question_selector = gr.Dropdown(
                choices=question_selector_choices, label="Question", container=False
            )

    model_selectors = [None] * num_sides
    with gr.Row():
        for i in range(num_sides):
            with gr.Column():
                model_selectors[i] = gr.Dropdown(
                    choices=models,
                    value=models[i] if len(models) > i else "",
                    label=f"Model {side_names[i]}",
                    container=False,
                )

    # Conversation
    chat_mds = []
    for i in range(num_turns):
        chat_mds.append(gr.Markdown(elem_id=f"user_question_{i+1}"))
        with gr.Row():
            for j in range(num_sides):
                with gr.Column(scale=100):
                    chat_mds.append(gr.Markdown())

                if j == 0:
                    with gr.Column(scale=1, min_width=8):
                        gr.Markdown()

    reference = gr.Markdown(elem_id=f"reference")
    chat_mds.append(reference)

    model_explanation = gr.Markdown(elem_id="model_explanation")
    model_explanation2 = gr.Markdown(elem_id="model_explanation")

    # Callbacks
    category_selector.change(display_question, [category_selector], [question_selector])
    question_selector.change(
        display_single_answer,
        [question_selector] + model_selectors,
        chat_mds + [model_explanation] + [model_explanation2],
    )

    for i in range(num_sides):
        model_selectors[i].change(
            display_single_answer,
            [question_selector] + model_selectors,
            chat_mds + [model_explanation] + [model_explanation2],
        )

    return (category_selector,)


block_css = """
#user_question_1 {
    background-color: #DEEBF7;
}
#user_question_2 {
    background-color: #E2F0D9;
}
#reference {
    background-color: #FFF2CC;
}
#model_explanation {
    background-color: #FBE5D6;
}
"""


def load_demo():
    dropdown_update = gr.Dropdown.update(value=list(category_selector_map.keys())[0])
    return dropdown_update, dropdown_update


def build_demo():
    build_question_selector_map()

    with gr.Blocks(
        title="MT-Bench Browser",
        theme=gr.themes.Base(text_size=gr.themes.sizes.text_lg),
        css=block_css,
    ) as demo:
        gr.Markdown(
            """
# MT-Bench Browser
The code to generate answers and judgments is at [fastchat.llm_judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge).
"""
        )
        with gr.Tab("Single Answer Grading"):
            (category_selector,) = build_single_answer_browser_tab()
        with gr.Tab("Pairwise Comparison"):
            (category_selector2,) = build_pairwise_browser_tab()
        demo.load(load_demo, [], [category_selector, category_selector2])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    args = parser.parse_args()
    print(args)

    question_file = f"data/{args.bench_name}/question.jsonl"
    answer_dir = f"data/{args.bench_name}/model_answer"
    pairwise_model_judgment_file = (
        f"data/{args.bench_name}/model_judgment/gpt-4_pair.jsonl"
    )
    single_model_judgment_file = (
        f"data/{args.bench_name}/model_judgment/gpt-4_single.jsonl"
    )

    # Load questions
    questions = load_questions(question_file, None, None)

    # Load answers
    model_answers = load_model_answers(answer_dir)

    # Load model judgments
    model_judgments_normal_single = (
        model_judgments_math_single
    ) = load_single_model_judgments(single_model_judgment_file)
    model_judgments_normal_pairwise = (
        model_judgments_math_pairwise
    ) = load_pairwise_model_judgments(pairwise_model_judgment_file)

    demo = build_demo()
    demo.queue(concurrency_count=10, status_update_rate=10, api_open=False).launch(
        server_name=args.host, server_port=args.port, share=args.share, max_threads=200
    )
