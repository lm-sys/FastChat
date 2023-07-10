"""
Add OpenAI moderation API results to all conversations.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time

import openai
import requests
from tqdm import tqdm


API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"


def tag_moderation(text):
    result = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            result = openai.Moderation.create(input=text)["results"][0]
            break
        except openai.error.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return result


def tag_openai_moderation(x):
    conv = x["conversation_a"]
    user_prompts = "\n".join([x["content"] for x in conv if x["role"] == "user"])
    result = tag_moderation(user_prompts)
    x["openai_moderation"] = result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument("--first-n", type=int)
    args = parser.parse_args()

    battles = json.load(open(args.input))

    if args.first_n:
        battles = battles[: args.first_n]

    with ThreadPoolExecutor(args.parallel) as executor:
        for line in tqdm(
            executor.map(tag_openai_moderation, battles), total=len(battles)
        ):
            pass

    output = args.input.replace(".json", "_tagged.json")
    with open(output, "w") as fout:
        json.dump(battles, fout, indent=2, ensure_ascii=False)
    print(f"Write cleaned data to {output}")
