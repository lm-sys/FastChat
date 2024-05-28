import argparse
import json
import pandas as pd
import os
import re
import ast
import time
import concurrent.futures
import tqdm
import random
import threading

LOCK = threading.RLock()

## Configs
SYSTEM_PROMPT = "Your task is to evaluate how well the following input prompts can assess the capabilities of advanced AI assistants.\n\nFor the input prompt, please analyze it based on the following 7 criteria.\n1. Specificity: Does the prompt ask for a specific output, such as code, a mathematical solution, a logical simplification, a problem-solving strategy, or a hardware setup recommendation? This specificity allows the AI to demonstrate its ability to understand and generate precise responses.\n2. Domain Knowledge: Does the prompt cover a specific domain, such as programming, mathematics, logic, problem-solving, or hardware setup? Prompts spanning a range of topics test the AI's breadth of knowledge and its ability to apply that knowledge to different domains.\n3. Complexity: Does the prompt vary in complexity, from straightforward tasks to more complex, multi-step problems? This allows evaluators to assess the AI's capability to handle problems of varying difficulty.\n4. Problem-Solving Skills: Does the prompt directly involves the AI to demonstrate active problem-solving skills, such systemically coming up with a solution for a specific setup instead of regurgitating an existing fact? This tests the AI's ability to apply logical reasoning and provide practical solutions.\n5. Creativity: Does the prompt involve a level of creativity in approaching the problem? This criterion tests the AI's ability to provide tailored solutions that take into account the user's specific needs and limitations.\n6. Technical Accuracy: Does the prompt require technical accuracy in the response? This allows evaluators to assess the AI's precision and correctness in technical fields.\n7. Real-world Application: Does the prompt relate to real-world applications, such as setting up a functional system or writing code for a practical use case? This tests the AI's ability to provide practical and actionable information that could be implemented in real-life scenarios.\n\nYou must list the criteria numbers that the prompt satisfies in the format of a Python array. For example, \"[...]\". Do not explain your choice."

ENDPOINT_INFO = {
    "model_name": "META-LLAMA/LLAMA-3-70B-CHAT-HF",
    "name": "llama-3-70b-instruct",
    "endpoints": [{"api_base": "-", "api_key": "-"}],
    "parallel": 8,
    "temperature": 0.0,
    "max_token": 512,
}  # Modify this

TAGS = {
    1: "specificity",
    2: "domain_knowledge",
    3: "complexity",
    4: "problem_solving",
    5: "creativity",
    6: "technical_accuracy",
    7: "real_world",
}

# API setting constants
API_MAX_RETRY = 3
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"


def get_endpoint(endpoint_list):
    if endpoint_list is None:
        return None
    assert endpoint_list is not None
    # randomly pick one
    api_dict = random.choices(endpoint_list)[0]
    return api_dict


pattern = re.compile(r"(\[\d(?:\,\s\d)*\])")


def get_score(judgment):
    matches = pattern.findall(judgment)
    matches = [m for m in matches if m != ""]
    if len(set(matches)) == 0:
        return []
    elif len(set(matches)) == 1:
        try:
            return ast.literal_eval(matches[0])
        except SyntaxError:
            print(matches[0])
            return []
    else:
        return []


def chat_completion_openai(model, messages, temperature, max_tokens, api_dict=None):
    import openai

    if api_dict:
        client = openai.OpenAI(
            base_url=api_dict["api_base"],
            api_key=api_dict["api_key"],
        )
    else:
        client = openai.OpenAI()

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            # print(messages)
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                # extra_body={"guided_choice": GUIDED_CHOICES} if GUIDED_CHOICES else None,
            )
            output = completion.choices[0].message.content
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(messages)
            print(type(e), e)
            break
        except openai.APIConnectionError as e:
            print(messages)
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.InternalServerError as e:
            print(messages)
            print(type(e), e)
            time.sleep(1)
        except KeyError:
            print(type(e), e)
            break

    return output


def get_answer(
    question: dict,
    max_tokens: int,
    temperature: float,
    answer_file: str,
    api_dict: dict,
):
    conv = []
    conv.append({"role": "system", "content": SYSTEM_PROMPT})

    conv.append({"role": "user", "content": question["prompt"]})
    output = chat_completion_openai(
        model=ENDPOINT_INFO["model_name"],
        messages=conv,
        temperature=temperature,
        max_tokens=max_tokens,
        api_dict=api_dict,
    )

    criteria = get_score(output)

    # Dump answers
    question["criteria_tag"] = {name: bool(i in criteria) for i, name in TAGS.items()}
    question.drop("prompt")

    with LOCK:
        with open(answer_file, "a") as fout:
            fout.write(json.dumps(question.to_dict()) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--cache-file", type=str, default=None)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--convert-to-json", action="store_true")
    args = parser.parse_args()

    print("loading input data (might take min)")
    input_data = pd.read_json(args.input_file)
    print(f"{len(input_data)}# of input data just loaded")
    if args.cache_file:
        print("loading cache data")
        cache_data = pd.read_json(args.cache_file)
        print(f"{len(cache_data)}# of cache data just loaded")

        assert "criteria_tag" in cache_data.columns and len(
            cache_data["criteria_tag"].dropna()
        ) == len(cache_data)

        not_labeled = input_data[
            ~input_data["question_id"].isin(cache_data["question_id"])
        ].copy()
    else:
        not_labeled = input_data.copy()

    if os.path.isfile(args.output_file):
        print("loading existing output")
        output_data = pd.read_json(args.output_file, lines=True)
        print(f"{len(output_data)}# of existing output just loaded")

        assert "criteria_tag" in output_data.columns and len(
            output_data["criteria_tag"].dropna()
        ) == len(output_data)

        not_labeled = not_labeled[
            ~not_labeled["question_id"].isin(output_data["question_id"])
        ]

    print(f"{len(not_labeled)} needs to be labeled")

    not_labeled["prompt"] = not_labeled.conversation_a.map(
        lambda convo: "\n".join([convo[i]["content"] for i in range(0, len(convo), 2)])
    )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=ENDPOINT_INFO["parallel"]
    ) as executor:
        futures = []
        for index, row in tqdm.tqdm(not_labeled.iterrows()):
            future = executor.submit(
                get_answer,
                row,
                ENDPOINT_INFO["max_token"],
                ENDPOINT_INFO["temperature"],
                args.output_file,
                get_endpoint(ENDPOINT_INFO["endpoints"]),
            )
            futures.append(future)
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()

    if args.convert_to_json:
        temp = pd.read_json(args.output_file, lines=True)
        temp.to_json(
            args.output_file[:-1], orient="records", indent=4, force_ascii=False
        )
