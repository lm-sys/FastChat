"""
Common data structures and utilities.
"""

import ast
import dataclasses
import glob
import json
import os
import re
import time
from typing import Optional

import openai
import anthropic

from fastchat.model.model_adapter import (
    get_conversation_template,
    ANTHROPIC_MODEL_LIST,
    OPENAI_MODEL_LIST,
)

# API setting constants
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"

TIE_DELTA = 0.1

# Categories that need reference answers
NEED_REF_CATS = ["math", "reasoning", "coding", "arena-hard-200"]

# Extract scores from judgments
two_score_pattern = re.compile("\[\[(\d+\.?\d*),\s?(\d+\.?\d*)\]\]")
two_score_pattern_backup = re.compile("\[(\d+\.?\d*),\s?(\d+\.?\d*)\]")
one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")

# Sampling temperature configs for
temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
    "arena-hard-200": 0.0,
}

reverse_model_map = {
    "model_1": "model_2",
    "model_2": "model_1",
}


@dataclasses.dataclass
class Judge:
    model_name: str
    prompt_template: dict
    ref_based: bool = False
    multi_turn: bool = False


@dataclasses.dataclass
class MatchSingle:
    question: dict
    model: str
    answer: dict
    judge: Judge
    ref_answer: dict = None
    multi_turn: bool = False


@dataclasses.dataclass
class MatchPair:
    question: dict
    model_1: str
    model_2: str
    answer_1: dict
    answer_2: dict
    judge: Judge
    ref_answer: dict = None
    multi_turn: bool = False


def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions


def load_model_answers(answer_dir: str):
    """Load model answers.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[question_id: int -> answer: dict]]
    """
    filenames = glob.glob(os.path.join(answer_dir, "*.jsonl"))
    filenames.sort()
    model_answers = {}

    for filename in filenames:
        model_name = os.path.basename(filename)[:-6]
        answer = {}
        with open(filename) as fin:
            for line in fin:
                line = json.loads(line)
                answer[line["question_id"]] = line
        model_answers[model_name] = answer

    return model_answers


def load_judge_prompts(prompt_file: str):
    """Load judge prompts.

    The return value is a python dict of type:
    Dict[judge_name: str -> dict]
    """
    prompts = {}
    with open(prompt_file) as fin:
        for line in fin:
            line = json.loads(line)
            prompts[line["name"]] = line
    return prompts


def run_judge_single(question, answer, judge, ref_answer, multi_turn=False):
    kwargs = {}
    model = judge.model_name
    if ref_answer is not None:
        kwargs["ref_answer_1"] = ref_answer["choices"][0]["turns"][0]
        if multi_turn:
            kwargs["ref_answer_2"] = ref_answer["choices"][0]["turns"][1]

    if multi_turn:
        user_prompt = judge.prompt_template["prompt_template"].format(
            question_1=question["turns"][0],
            question_2=question["turns"][1],
            answer_1=answer["choices"][0]["turns"][0],
            answer_2=answer["choices"][0]["turns"][1],
            **kwargs,
        )
    else:
        user_prompt = judge.prompt_template["prompt_template"].format(
            question=question["turns"][0],
            answer=answer["choices"][0]["turns"][0],
            **kwargs,
        )

    rating = -1

    system_prompt = judge.prompt_template["system_prompt"]
    conv = get_conversation_template(model)
    conv.set_system_message(system_prompt)
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)

    if model in OPENAI_MODEL_LIST:
        judgment = chat_completion_openai(model, conv, temperature=0, max_tokens=2048)
    elif model in ANTHROPIC_MODEL_LIST:
        judgment = chat_completion_anthropic(
            model, conv, temperature=0, max_tokens=1024
        )
    else:
        raise ValueError(f"Invalid judge model name: {model}")

    if judge.prompt_template["output_format"] == "[[rating]]":
        match = re.search(one_score_pattern, judgment)
        if not match:
            match = re.search(one_score_pattern_backup, judgment)

        if match:
            rating = ast.literal_eval(match.groups()[0])
        else:
            rating = -1
    else:
        raise ValueError(
            f"invalid output format: {judge.prompt_template['output_format']}"
        )

    return rating, user_prompt, judgment


def play_a_match_single(match: MatchPair, output_file: str):
    question, model, answer, judge, ref_answer, multi_turn = (
        match.question,
        match.model,
        match.answer,
        match.judge,
        match.ref_answer,
        match.multi_turn,
    )

    if judge.prompt_template["type"] == "single":
        score, user_prompt, judgment = run_judge_single(
            question, answer, judge, ref_answer, multi_turn=multi_turn
        )

        question_id = question["question_id"]
        turn = 1 if not multi_turn else 2
        result = {
            "question_id": question_id,
            "model": model,
            "judge": (judge.model_name, judge.prompt_template["name"]),
            "user_prompt": user_prompt,
            "judgment": judgment,
            "score": score,
            "turn": turn,
            "tstamp": time.time(),
        }
        print(
            f"question: {question_id}, turn: {turn}, model: {model}, "
            f"score: {score}, "
            f"judge: {(judge.model_name, judge.prompt_template['name'])}"
        )
    else:
        raise ValueError(f"invalid judge type: {judge['type']}")

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a") as fout:
            fout.write(json.dumps(result) + "\n")

    return result


def run_judge_pair(question, answer_a, answer_b, judge, ref_answer, multi_turn=False):
    kwargs = {}
    model = judge.model_name
    if ref_answer is not None:
        kwargs["ref_answer_1"] = ref_answer["choices"][0]["turns"][0]
        if multi_turn:
            kwargs["ref_answer_2"] = ref_answer["choices"][0]["turns"][1]

    if multi_turn:
        system_prompt = judge.prompt_template["system_prompt"]
        user_prompt = judge.prompt_template["prompt_template"].format(
            question_1=question["turns"][0],
            question_2=question["turns"][1],
            answer_a_1=answer_a["choices"][0]["turns"][0],
            answer_b_1=answer_b["choices"][0]["turns"][0],
            answer_a_2=answer_a["choices"][0]["turns"][1],
            answer_b_2=answer_b["choices"][0]["turns"][1],
            **kwargs,
        )
    else:
        system_prompt = judge.prompt_template["system_prompt"]
        user_prompt = judge.prompt_template["prompt_template"].format(
            question=question["turns"][0],
            answer_a=answer_a["choices"][0]["turns"][0],
            answer_b=answer_b["choices"][0]["turns"][0],
            **kwargs,
        )

    winner = "error"

    conv = get_conversation_template(model)
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)

    if model in OPENAI_MODEL_LIST:
        conv.set_system_message(system_prompt)
        judgment = chat_completion_openai(model, conv, temperature=0, max_tokens=2048)
    elif model in ANTHROPIC_MODEL_LIST:
        if system_prompt != "You are a helpful assistant.":
            user_prompt = "[Instruction]\n" + system_prompt + "\n\n" + user_prompt
            conv.messages[0][1] = user_prompt
        judgment = chat_completion_anthropic(
            model, conv, temperature=0, max_tokens=1024
        )
    else:
        raise ValueError(f"Invalid judge model name: {model}")

    if judge.prompt_template["output_format"] == "[[A]]":
        if "[[A]]" in judgment:
            winner = "A"
        elif "[[B]]" in judgment:
            winner = "B"
        elif "[[C]]" in judgment:
            winner = "tie"
        else:
            winner = "error"
    elif judge.prompt_template["output_format"] == "[[rating_a,rating_b]]":
        match = re.search(two_score_pattern, judgment)
        if not match:
            match = re.search(two_score_pattern_backup, judgment)
        if match:
            scores = [ast.literal_eval(s.strip()) for s in match.groups()]
            if abs(scores[0] - scores[1]) <= TIE_DELTA:
                winner = "tie"
            elif scores[0] > scores[1]:
                winner = "A"
            else:
                winner = "B"
        else:
            winner = "error"
    else:
        raise ValueError(
            f"invalid output format: {judge.prompt_template['output_format']}"
        )

    return winner, user_prompt, judgment


def play_a_match_pair(match: MatchPair, output_file: str):
    question, model_1, model_2, answer_1, answer_2, judge, ref_answer, multi_turn = (
        match.question,
        match.model_1,
        match.model_2,
        match.answer_1,
        match.answer_2,
        match.judge,
        match.ref_answer,
        match.multi_turn,
    )

    if judge.prompt_template["type"] == "pairwise":
        g1_winner, g1_user_prompt, g1_judgment = run_judge_pair(
            question, answer_1, answer_2, judge, ref_answer, multi_turn=multi_turn
        )
        g2_winner, g2_user_prompt, g2_judgment = run_judge_pair(
            question, answer_2, answer_1, judge, ref_answer, multi_turn=multi_turn
        )

        g1_map = {"A": "model_1", "B": "model_2"}
        g2_map = {"A": "model_2", "B": "model_1"}
        g1_winner = g1_map.get(g1_winner, g1_winner)
        g2_winner = g2_map.get(g2_winner, g2_winner)
        question_id = question["question_id"]
        turn = 1 if not multi_turn else 2

        result = {
            "question_id": question_id,
            "model_1": model_1,
            "model_2": model_2,
            "g1_winner": g1_winner,
            "g2_winner": g2_winner,
            "judge": (judge.model_name, judge.prompt_template["name"]),
            "g1_user_prompt": g1_user_prompt,
            "g1_judgment": g1_judgment,
            "g2_user_prompt": g2_user_prompt,
            "g2_judgment": g2_judgment,
            "turn": turn,
            "tstamp": time.time(),
        }

        print(
            f"question: {question_id}, turn: {turn}, model_1: {model_1}, model_2: {model_2}, "
            f"g1_winner: {g1_winner}, g2_winner: {g2_winner}, "
            f"judge: {(judge.model_name, judge.prompt_template['name'])}"
        )
    elif judge.prompt_template["type"] == "single":
        m1_score, m1_user_prompt, m1_judgment = run_judge_single(
            question, answer_1, judge
        )
        m2_score, m2_user_prompt, m2_judgment = run_judge_single(
            question, answer_2, judge
        )

        if abs(m1_score - m2_score) <= TIE_DELTA:
            winner = "tie"
        elif m1_score > m2_score:
            winner = "model_1"
        else:
            winner = "model_2"

        question_id = question["question_id"]
        result = {
            "question_id": question_id,
            "model_1": model_1,
            "model_2": model_2,
            "g1_winner": winner,
            "g2_winner": winner,
            "judge": (judge.model_name, judge.prompt_template["name"]),
            "g1_user_prompt": m1_user_prompt,
            "g1_judgment": m1_judgment,
            "g2_user_prompt": m2_user_prompt,
            "g2_judgment": m2_judgment,
            "m1_score": m1_score,
            "m2_score": m2_score,
            "tstamp": time.time(),
        }
        print(
            f"question: {question_id}, model_1: {model_1}, model_2: {model_2}, "
            f"winner: {winner}, m1_score: {m1_score}, m2_score: {m2_score}, "
            f"judge: {(judge.model_name, judge.prompt_template['name'])}"
        )
    else:
        raise ValueError(f"invalid judge type: {judge['type']}")

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a") as fout:
            fout.write(json.dumps(result) + "\n")

    return result


def chat_completion_openai(model, conv, temperature, max_tokens, api_dict=None):
    if api_dict is not None:
        openai.api_base = api_dict["api_base"]
        openai.api_key = api_dict["api_key"]
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response["choices"][0]["message"]["content"]
            break
        except openai.error.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return output


def chat_completion_openai_azure(model, conv, temperature, max_tokens, api_dict=None):
    openai.api_type = "azure"
    openai.api_version = "2023-07-01-preview"
    if api_dict is not None:
        openai.api_base = api_dict["api_base"]
        openai.api_key = api_dict["api_key"]
    else:
        openai.api_base = os.environ["AZURE_OPENAI_ENDPOINT"]
        openai.api_key = os.environ["AZURE_OPENAI_KEY"]

    if "azure-" in model:
        model = model[6:]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            response = openai.ChatCompletion.create(
                engine=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response["choices"][0]["message"]["content"]
            break
        except openai.error.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.error.InvalidRequestError as e:
            print(type(e), e)
            break
        except KeyError:
            print(response)
            break

    return output


def chat_completion_anthropic(model, conv, temperature, max_tokens, api_dict=None):
    if api_dict is not None and "api_key" in api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["ANTHROPIC_API_KEY"]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            c = anthropic.Anthropic(api_key=api_key)
            prompt = conv.get_prompt()
            response = c.completions.create(
                model=model,
                prompt=prompt,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                max_tokens_to_sample=max_tokens,
                temperature=temperature,
            )
            output = response.completion
            break
        except anthropic.APIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output.strip()


def chat_completion_palm(chat_state, model, conv, temperature, max_tokens):
    from fastchat.serve.api_provider import init_palm_chat

    assert model == "palm-2-chat-bison-001"

    if chat_state is None:
        chat_state = init_palm_chat("chat-bison@001")

    parameters = {
        "temperature": temperature,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": max_tokens,
    }
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = chat_state.send_message(conv.messages[-2][1], **parameters)
            output = response.text
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return chat_state, output


def normalize_game_key_single(gamekey, result):
    """Make the model names sorted in a game key."""
    qid, model_1, model_2 = gamekey
    if model_1 < model_2:
        return gamekey, result
    else:
        new_gamekey = (qid, model_2, model_1)
        new_result = {
            "winners": tuple(reverse_model_map.get(x, x) for x in result["winners"]),
            "g1_judgment": result["g2_judgment"],
            "g2_judgment": result["g1_judgment"],
        }
        return new_gamekey, new_result


def normalize_game_key_dict(judgment_dict):
    """Make the model names sorted in the game keys."""
    ret = {}
    for key, value in judgment_dict.items():
        new_key, new_value = normalize_game_key_single(key, value)
        ret[new_key] = new_value
    return ret


def load_pairwise_model_judgments(filename: str):
    """Load model judgments.

    The return value is a dict of type:
    Dict[judge: Tuple -> Dict[game_key: tuple -> game_result: dict]
    """
    judge_dict = {}

    for line in open(filename):
        obj = json.loads(line)
        judge = tuple(obj["judge"])
        qid, model_1, model_2 = obj["question_id"], obj["model_1"], obj["model_2"]

        if judge not in judge_dict:
            judge_dict[judge] = {}

        if "winner" in obj:
            winner = obj["winner"]
        elif "g1_winner" in obj and "g2_winner" in obj:
            g1_winner, g2_winner = obj["g1_winner"], obj["g2_winner"]
            if g1_winner == g2_winner:
                winner = g1_winner
            else:
                winner = "inconsistent"
        else:
            raise ValueError(f"Invalid keys: {list(obj.keys())}")

        gamekey = (qid, model_1, model_2)
        winners = (winner,)

        judge_dict[judge][gamekey] = {
            "winners": winners,
            "g1_judgment": obj["g1_judgment"],
            "g2_judgment": obj["g2_judgment"],
        }

    # Make the model names sorted in the game keys
    normalized = {}
    for judge, value in judge_dict.items():
        normalized[judge] = normalize_game_key_dict(value)
    return normalized


def load_single_model_judgments(filename: str):
    """Load model judgments.

    The return value is a dict of type:
    Dict[judge: Tuple -> Dict[game_key: tuple -> game_result: dict]
    """
    judge_dict = {}

    for line in open(filename):
        obj = json.loads(line)
        judge = tuple(obj["judge"])
        qid, model = obj["question_id"], obj["model"]

        if judge not in judge_dict:
            judge_dict[judge] = {}

        gamekey = (qid, model)

        judge_dict[judge][gamekey] = {
            "score": obj["score"],
            "judgment": obj["judgment"],
        }
    return judge_dict


def resolve_pairwise_judgment_dict(
    question, model_judgments_normal, model_judgments_math, multi_turn=False
):
    """Return the correct pairwise judge."""
    if multi_turn:
        if question["category"] in NEED_REF_CATS:
            return model_judgments_math[("gpt-4", "pair-math-v1-multi-turn")]
        return model_judgments_normal[("gpt-4", "pair-v2-multi-turn")]

    if question["category"] in NEED_REF_CATS:
        return model_judgments_math[("gpt-4", "pair-math-v1")]
    else:
        return model_judgments_normal[("gpt-4", "pair-v2")]


def resolve_single_judgment_dict(
    question, model_judgments_normal, model_judgments_math, multi_turn=False
):
    """Return the correct single answer grading judge."""
    if multi_turn:
        if question["category"] in NEED_REF_CATS:
            return model_judgments_math[("gpt-4", "single-math-v1-multi-turn")]
        return model_judgments_normal[("gpt-4", "single-v1-multi-turn")]

    if question["category"] in NEED_REF_CATS:
        return model_judgments_math[("gpt-4", "single-math-v1")]
    else:
        return model_judgments_normal[("gpt-4", "single-v1")]


def get_pairwise_judge_explanation(gamekey, judgment_dict):
    """Get model judge explanation."""
    try:
        qid, model_1, model_2 = gamekey
        if model_1 < model_2:
            res = judgment_dict[gamekey]
            g1_judgment, g2_judgment = res["g1_judgment"], res["g2_judgment"]
        else:
            new_gamekey = (qid, model_2, model_1)
            res = judgment_dict[new_gamekey]

            model_1, model_2 = model_1, model_2
            g1_judgment, g2_judgment = res["g2_judgment"], res["g1_judgment"]

        return (
            f"**Game 1**. **A**: {model_1}, **B**: {model_2}\n\n"
            f"**Judgment**: {g1_judgment}"
            + f"\n\n`--------------------------`\n\n"
            + f"**Game 2**. **A**: {model_2}, **B**: {model_1}\n\n"
            f"**Judgment**: {g2_judgment}"
        )
    except KeyError:
        return "N/A"


def get_single_judge_explanation(gamekey, judgment_dict):
    """Get model judge explanation."""
    try:
        qid, model = gamekey

        res = judgment_dict[gamekey]

        g1_judgment = res["judgment"]
        g1_score = res["score"]

        return (
            f"**Game 1**. **A**: {model}, **Score**: {g1_score}\n\n"
            f"**Judgment**: {g1_judgment}"
        )
    except KeyError:
        return "N/A"


def check_data(questions, model_answers, ref_answers, models, judges):
    # check model answers
    for m in models:
        assert m in model_answers, f"Missing model answer for {m}"
        m_answer = model_answers[m]
        for q in questions:
            assert (
                q["question_id"] in m_answer
            ), f"Missing model {m}'s answer to Question {q['question_id']}"
    # check ref answers
    for jg in judges.values():
        if not jg.ref_based:
            continue
        for q in questions:
            if q["category"] not in NEED_REF_CATS:
                continue
            assert (
                q["question_id"] in ref_answers[jg.model_name]
            ), f"Missing reference answer to Question {q['question_id']} for judge {jg.model_name}"


def get_model_list(answer_dir):
    file_paths = glob.glob(f"{answer_dir}/*.jsonl")
    file_names = [os.path.splitext(os.path.basename(f))[0] for f in file_paths]
    return file_names
