import numpy as np
import wandb

import argparse
import json
import os
import random
import time
from types import SimpleNamespace

import hashlib
import datetime

import shortuuid
import torch
from tqdm import tqdm

# default configs
default_config  = SimpleNamespace(
    # for gen_model_answer
    model_path='Open-Orca/Mistral-7B-OpenOrca',
    model_id=None,
    bench_name='japanese_mt_bench',
    question_begin=None,
    question_end=None,
    answer_file=None,
    max_new_token=1024,
    num_choices=1,
    num_gpus_per_model=1,
    num_gpus_total=1,
    max_gpu_memory=None,
    dtype=None,
    # for gen_judgment
    judge_file="fastchat/llm_judge/data/judge_prompts.jsonl",
    judge_model="gpt-4",
    baseline_model="gpt-3.5-turbo",
    mode="single",
    model_list=None,
    parallel=1,
    first_n=None,
    # for conv template # added
    custom_conv_template=False,
    conv_name="custom",
    conv_system_message="以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。",
    conv_roles="('指示', '応答')",
    conv_sep="\n\n### ",
    conv_stop_token_ids="[2]",
    conv_stop_str="###",
    conv_role_message_separator=": \n",
    conv_role_only_separator=": \n",
)

def evaluate(run_id=None, config=default_config):

    # create hash and append it to the model_id in order to avoid duplicated id
    mnaum_data = str(datetime.datetime.now())
    encoded_data = mnaum_data.encode()
    hash_object = hashlib.sha256(encoded_data)
    hashed_string = hash_object.hexdigest()
    
    if config.model_id == None:
        config.model_id = f'{config.model_path.replace("/", "--")}_hash_{hashed_string}'


    # initialize wandb run
    if run_id==None:
        run = wandb.init(project='yuya-test-llm', config=config)
        config = run.config
    else:
        run = wandb.init(project='yuya-test-llm', id=run_id, resume="allow")
        config = run.config
        
    from fastchat.llm_judge.common import load_questions, temperature_config
    from fastchat.model import load_model, get_conversation_template
    from fastchat.utils import str_to_torch_dtype

    from fastchat.llm_judge.gen_model_answer import get_conversation_template, get_model_answers, load_model, load_questions, reorg_answer_file, run_eval, str_to_torch_dtype
    from fastchat.llm_judge.gen_judgment import check_data, get_model_list, load_judge_prompts, load_model_answers, load_questions, make_judge_pairwise, make_judge_single, make_match, make_match_all_pairs, make_match_single, play_a_match_pair, play_a_match_single
    

    if config.num_gpus_total // config.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"fastchat/llm_judge/data/{config.bench_name}/question.jsonl"
    if config.answer_file:
        answer_file = config.answer_file
    else:
        answer_file = f"fastchat/llm_judge/data/{config.bench_name}/model_answer/{config.model_id}.jsonl"

    print(f"Output to {answer_file}")

    # 1. generate model answers

    run_eval(
        model_path=config.model_path,
        model_id=config.model_id,
        question_file=question_file,
        question_begin=config.question_begin,
        question_end=config.question_end,
        answer_file=answer_file,
        max_new_token=config.max_new_token,
        num_choices=config.num_choices,
        num_gpus_per_model=config.num_gpus_per_model,
        num_gpus_total=config.num_gpus_total,
        max_gpu_memory=config.max_gpu_memory,
        dtype=str_to_torch_dtype(config.dtype),
    )


    # 2. evaluate outputs

    import argparse
    from concurrent.futures import ThreadPoolExecutor
    import json

    import numpy as np
    from tqdm import tqdm

    from fastchat.llm_judge.common import (
        load_questions,
        load_model_answers,
        load_judge_prompts,
        check_data,
        play_a_match_pair,
        play_a_match_single,
        get_model_list,
        Judge,
        MatchPair,
        MatchSingle,
        NEED_REF_CATS,
    )

    import openai
    openai.api_key = os.environ["OPENAI_API_KEY"]

    ## file path
    question_file = f"fastchat/llm_judge/data/{config.bench_name}/question.jsonl"
    answer_dir = f"fastchat/llm_judge/data/{config.bench_name}/model_answer"
    ref_answer_dir = f"fastchat/llm_judge/data/{config.bench_name}/reference_answer"

    ## Load questions
    questions = load_questions(question_file, None, None)

    ## Load answers
    model_answers = load_model_answers(answer_dir)
    model_answers = {config.model_id: model_answers[config.model_id]}
    ref_answers = load_model_answers(ref_answer_dir)

    ## Load judge
    judge_prompts = load_judge_prompts(config.judge_file)

    if config.first_n:
        questions = questions[: config.first_n]

    if config.model_list is None:
        models = [config.model_id] #get_model_list(answer_dir)
    else:
        models = config.model_list

    if config.mode == "single":
        judges = make_judge_single(config.judge_model, judge_prompts)
        play_a_match_func = play_a_match_single
        output_file = (
            f"fastchat/llm_judge/data/{config.bench_name}/model_judgment/{config.judge_model}_single.jsonl"
        )
        make_match_func = make_match_single
        baseline_model = None
    else:
        judges = make_judge_pairwise(config.judge_model, judge_prompts)
        play_a_match_func = play_a_match_pair
        output_file = (
            f"fastchat/llm_judge/data/{config.bench_name}/model_judgment/{config.judge_model}_pair.jsonl"
        )
        if config.mode == "pairwise-all":
            make_match_func = make_match_all_pairs
            baseline_model = None
        else:
            make_match_func = make_match
            baseline_model = config.baseline_model

    check_data(questions, model_answers, ref_answers, models, judges)

    question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
    question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

    # Make matches
    matches = []
    matches += make_match_func(
        question_default, models, model_answers, judges["default"], baseline_model
    )
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math"],
        baseline_model,
        ref_answers,
    )
    matches += make_match_func(
        question_default,
        models,
        model_answers,
        judges["default-mt"],
        baseline_model,
        multi_turn=True,
    )
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math-mt"],
        baseline_model,
        ref_answers,
        multi_turn=True,
    )

    match_stat = {}
    match_stat["bench_name"] = config.bench_name
    match_stat["mode"] = config.mode
    match_stat["judge"] = config.judge_model
    match_stat["baseline"] = baseline_model
    match_stat["model_list"] = models
    match_stat["total_num_questions"] = len(questions)
    match_stat["total_num_matches"] = len(matches)
    match_stat["output_path"] = output_file

    # Show match stats and prompt enter to continue
    print("Stats:")
    print(json.dumps(match_stat, indent=4))
    #input("Press Enter to confirm...")

    # Play matches
    if config.parallel == 1:
        for match in tqdm(matches):
            play_a_match_func(match, output_file=output_file)
    else:

        def play_a_match_wrapper(match):
            play_a_match_func(match, output_file=output_file)

        np.random.seed(0)
        np.random.shuffle(matches)

        with ThreadPoolExecutor(config.parallel) as executor:
            for match in tqdm(
                executor.map(play_a_match_wrapper, matches), total=len(matches)
            ):
                pass

    # 3. consolidate results and log as wandb.Table

    import numpy as np
    import pandas as pd

    # load questions
    df_question = pd.read_json(question_file, lines=True)

    # load answers
    df_answer = pd.read_json(f"fastchat/llm_judge/data/{config.bench_name}/model_answer/{config.model_id}.jsonl", lines=True)
    df_answer = df_answer[df_answer.model_id == config.model_id]
    df_answer = df_answer.sort_values(['question_id'])

    # load judge results
    df_judge = pd.read_json(output_file, lines=True)
    df_judge = df_judge[df_judge.model == config.model_id]
    df_judge.model = df_judge.model.str.replace("--", "/")
    df_judge['hash'] = df_judge.model.apply(lambda x: x.split('_hash_')[-1])
    df_judge['model'] = df_judge.model.apply(lambda x: x.split('_hash_')[0])
    df_judge = df_judge.sort_values(['question_id', 'turn'])

    ## merge tables
    df_judge["question"] = np.nan
    df_judge.loc[df_judge.turn==1, 'question'] = df_question.turns.apply(lambda x: x[0]).values
    df_judge.loc[df_judge.turn==2, 'question'] = df_question.turns.apply(lambda x: x[1]).values

    df_judge['answer'] = np.nan
    df_judge.loc[df_judge.turn==1, 'answer'] = df_answer.choices.apply(lambda x: x[0][ 'turns'][0]).values
    df_judge.loc[df_judge.turn==2, 'answer'] = df_answer.choices.apply(lambda x: x[0][ 'turns'][1]).values
    df_judge = df_judge.merge(df_answer[['question_id', 'answer_id']], on='question_id', how='left')
    df_judge = df_judge.merge(df_question[['question_id', 'category']], on='question_id', how='left')

    ## clean dataframe up
    use_col = [
        'question_id', 'category', 'answer_id', 'model', 'question', 
        'answer', 'judge', 'user_prompt', 'judgment', 
        'score', 'turn', 'tstamp'
    ]
    df_judge = df_judge[use_col]

    table_log = wandb.Table(dataframe=df_judge)

    # table for radar chart
    df_summary = df_judge.groupby(['category'], as_index=False).score.mean()
    table_radar = wandb.Table(dataframe=df_summary)
    
    ## table for LB
    columns = ['model_name'] + df_summary.category.values.tolist()
    data = [[config.model_id.replace("--", "/").split('_hash_')[0]] + df_summary.score.values.tolist()]
    table_metric = wandb.Table(data=data, columns=columns)

    run.log({
        "log_table":table_log,
        "metric_table":table_metric,
        "radar_table":table_radar,
    })

    run.finish()

    return run_id

if __name__ == "__main__":
    evaluate()
