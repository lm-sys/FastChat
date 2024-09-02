import argparse
import json
import pandas as pd
import os
import time
import concurrent.futures
import tqdm
import yaml
import random
import threading
import orjson
import hashlib

from category import Category
from vision_utils import get_image_path

import lmdb

if not os.path.exists("cache/category_cache"):
    os.makedirs("cache/category_cache")
category_cache = lmdb.open("cache/category_cache", map_size=1024 ** 4)


LOCK = threading.RLock()

TASKS = None
CACHE_DICT = None
OUTPUT_DICT = None

# API setting constants
API_MAX_RETRY = None
API_RETRY_SLEEP = None
API_ERROR_OUTPUT = None


# load config args from config yaml files
def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)
    return config_kwargs


def get_endpoint(endpoint_list):
    if endpoint_list is None:
        return None
    assert endpoint_list is not None
    # randomly pick one
    api_dict = random.choices(endpoint_list)[0]
    return api_dict


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
            print(output)
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
            time.sleep(API_RETRY_SLEEP)
        except Exception as e:
            print(type(e), e)
            break

    return output


def get_answer(
    question: dict,
    model_name: str,
    max_tokens: int,
    temperature: float,
    answer_file: str,
    api_dict: dict,
    categories: list,
    testing: bool,
):
    if "category_tag" in question:
        category_tag = question["category_tag"]
    else:
        category_tag = {}

    output_log = {}

    for category in categories:
        conv = category.pre_process(question)
        output = chat_completion_openai(
            model=model_name,
            messages=conv,
            temperature=temperature,
            max_tokens=max_tokens,
            api_dict=api_dict,
        )
        # Dump answers
        category_tag[category.name_tag] = category.post_process(output)

        if testing:
            output_log[category.name_tag] = output

    question["category_tag"] = category_tag
    if testing:
        question["output_log"] = output_log

    # question.drop(["prompt", "uid", "required_tasks"], inplace=True)

    with LOCK:
        with open(answer_file, "a") as fout:
            fout.write(json.dumps(question.to_dict()) + "\n")

# def series_to_dict(obj):
#     if isinstance(obj, pd.Series):
#         return obj.to_dict()
#     elif isinstance(obj, dict):
#         return {k: series_to_dict(v) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         return [series_to_dict(item) for item in obj]
#     else:
#         print(f"Unknown type: {type(obj)}")
#         return obj

# def generate_cache_key(question, model_name, max_tokens, temperature):
#     cache_key_data = {
#         "question": series_to_dict(question),
#         "model_name": model_name,
#         "max_tokens": max_tokens,
#         "temperature": temperature,
#     }
#     return hashlib.md5(json.dumps(cache_key_data, sort_keys=True).encode()).hexdigest()

# def load_from_cache(cache, cache_key):
#     with cache.begin(write=False) as txn:
#         cached_result = txn.get(cache_key.encode())
#         if cached_result:
#             return json.loads(cached_result.decode())
#     return None

# def save_to_cache(cache, cache_key, data):
#     with cache.begin(write=True) as txn:
#         txn.put(cache_key.encode(), json.dumps(data).encode())

# def get_answer(
#     question: dict,
#     model_name: str,
#     max_tokens: int,
#     temperature: float,
#     answer_file: str,
#     api_dict: dict,
#     categories: list,
#     testing: bool,
#     cache: bool,
# ):
#     if "category_tag" in question:
#         category_tag = question["category_tag"]
#     else:
#         category_tag = {}

#     output_log = {}

#     cache_key = generate_cache_key(question, model_name, max_tokens, temperature)

#     if cache:
#         cached_data = load_from_cache(category_cache, cache_key)
#         if cached_data:
#             question["category_tag"] = cached_data["category_tag"]
#             if testing:
#                 question["output_log"] = cached_data["output_log"]

#     if not cache or not cached_data:
#         for category in categories:
#             conv = category.pre_process(question)
#             output = chat_completion_openai(
#                 model=model_name,
#                 messages=conv,
#                 temperature=temperature,
#                 max_tokens=max_tokens,
#                 api_dict=api_dict,
#             )
#             # Dump answers
#             category_tag[category.name_tag] = category.post_process(output)

#             if testing:
#                 output_log[category.name_tag] = output

#         question["category_tag"] = category_tag
#         if testing:
#             question["output_log"] = output_log

#     if cache:
#         save_to_cache(category_cache, cache_key, {
#             "category_tag": category_tag,
#             "output_log": output_log if testing else {}
#         })

#     with LOCK:
#         with open(answer_file, "a") as fout:
#             fout.write(json.dumps(series_to_dict(question)) + "\n")


def category_merge(row):
    id = row["uid"]
    input_category = row["category_tag"] if "category_tag" in row else {}
    cache_category = CACHE_DICT[id]["category_tag"] if id in CACHE_DICT else {}
    output_category = OUTPUT_DICT[id]["category_tag"] if id in OUTPUT_DICT else {}

    # tries to fill in missing categories using cache first, then output
    for name in TASKS:
        if name not in input_category:
            if name in cache_category:
                input_category[name] = cache_category[name]
                continue
            if name in output_category:
                input_category[name] = output_category[name]

    return input_category


def find_required_tasks(row):
    id = row["uid"]
    input_category = row["category_tag"] if "category_tag" in row else {}
    cache_category = CACHE_DICT[id]["category_tag"] if id in CACHE_DICT else {}
    output_category = OUTPUT_DICT[id]["category_tag"] if id in OUTPUT_DICT else {}

    return [
        name
        for name in TASKS
        if not (
            name in input_category or name in cache_category or name in output_category
        )
    ]

import wandb
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--cache", action="store_true")
    args = parser.parse_args()

    enter = input(
        "Make sure your config file is properly configured. Press enter to continue."
    )
    if not enter == "":
        exit()

    config = make_config(args.config)

    if not args.wandb:
        os.environ["WANDB_MODE"] = "dryrun"
    if args.wandb:
        wandb.init(project="arena", entity="clipinvariance", name=config["input_file"].split("/")[-1].split(".")[0])

    API_MAX_RETRY = config["max_retry"]
    API_RETRY_SLEEP = config["retry_sleep"]
    API_ERROR_OUTPUT = config["error_output"]

    categories = [Category.create_category(name) for name in config["task_name"]]
    TASKS = config["task_name"]
    print(
        f"Following categories will be labeled:\n{[category.name_tag for category in categories]}"
    )

    print("loading input data (might take min)")
    with open(config["input_file"], "rb") as f:
        data = orjson.loads(f.read())
    input_data = pd.DataFrame(data)

    if args.vision:
        old_len = len(input_data)
        input_data["image_hash"] = input_data.conversation_a.map(lambda convo: convo[0]["content"][1][0])
        input_data["image_path"] = input_data.image_hash.map(get_image_path)
        input_data = input_data[input_data.image_path != False].reset_index(drop=True)
        print(f"{len(input_data)} out of {old_len}# images found")

    if args.testing:
        if os.path.isfile(config["output_file"]):
            os.remove(config["output_file"])
        if "category_tag" in input_data.columns:
            input_data.drop(columns=["category_tag"], inplace=True)
        input_data = input_data[input_data['language'] == 'English'].reset_index(drop=True)
        input_data = input_data[:100]

    # much faster than pd.apply
    input_data["uid"] = input_data.question_id.map(str) + input_data.tstamp.map(str)
    assert len(input_data) == len(input_data.uid.unique())
    print(f"{len(input_data)}# of input data just loaded")

    if config["cache_file"]:
        print("loading cache data")
        with open(config["cache_file"], "rb") as f:
            data = orjson.loads(f.read())
        cache_data = pd.DataFrame(data)
        cache_data["uid"] = cache_data.question_id.map(str) + cache_data.tstamp.map(str)
        assert len(cache_data) == len(cache_data.uid.unique())

        print(f"{len(cache_data)}# of cache data just loaded")

        assert "category_tag" in cache_data.columns
        cache_dict = cache_data[["uid", "category_tag"]].set_index("uid")
        print("finalizing cache_dict (should take less than 30 sec)")
        CACHE_DICT = cache_dict.to_dict("index")
    else:
        CACHE_DICT = {}

    if os.path.isfile(config["output_file"]):
        print("loading existing output")
        output_data = pd.read_json(config["output_file"], lines=True)
        output_data["uid"] = output_data.question_id.map(str) + output_data.tstamp.map(
            str
        )
        assert len(output_data) == len(output_data.uid.unique())

        print(f"{len(output_data)}# of existing output just loaded")

        assert "category_tag" in output_data.columns
        output_dict = output_data[["uid", "category_tag"]].set_index("uid")
        print("finalizing output_dict (should take less than 30 sec)")
        OUTPUT_DICT = output_dict.to_dict("index")
    else:
        OUTPUT_DICT = {}

    print(
        "finding tasks needed to run... (should take around 1 minute or less on large dataset)"
    )
    input_data["required_tasks"] = input_data.apply(find_required_tasks, axis=1)

    not_labeled = input_data[input_data.required_tasks.map(lambda x: len(x) > 0)].copy()

    print(f"{len(not_labeled)} # of conversations needs to be labeled")
    for name in TASKS:
        print(
            f"{name}: {len(not_labeled[not_labeled.required_tasks.map(lambda tasks: name in tasks)])}"
        )

    get_content = lambda c: c if type(c) == str else c[0]
    not_labeled["prompt"] = not_labeled.conversation_a.map(
        lambda convo: "\n".join([get_content(convo[i]["content"]) for i in range(0, len(convo), 2)])
    )
    not_labeled["prompt"] = not_labeled.prompt.map(lambda x: x[:12500])
    not_labeled["response_a"] = not_labeled.conversation_a.map(
        lambda convo: "\n".join([get_content(convo[i]["content"]) for i in range(1, len(convo), 2)])
    )
    not_labeled["response_a"] = not_labeled.response_a.map(lambda x: x[:12500])
    not_labeled["response_b"] = not_labeled.conversation_b.map(
        lambda convo: "\n".join([get_content(convo[i]["content"]) for i in range(1, len(convo), 2)])
    )
    not_labeled["response_b"] = not_labeled.response_b.map(lambda x: x[:12500])

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config["parallel"]
    ) as executor:
        futures = []
        for index, row in tqdm.tqdm(not_labeled.iterrows()):
            future = executor.submit(
                get_answer,
                row,
                config["model_name"],
                config["max_token"],
                config["temperature"],
                config["output_file"],
                get_endpoint(config["endpoints"]),
                [
                    category
                    for category in categories
                    if category.name_tag in row["required_tasks"]
                ],
                args.testing,
                # args.cache,
            )
            futures.append(future)
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()

    output = pd.read_json(config["output_file"], lines=True)
        
    # log table to wandb
    if args.wandb:
        def replace_none_in_nested_dict(d):
            if isinstance(d, dict):
                return {k: replace_none_in_nested_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [replace_none_in_nested_dict(v) for v in d]
            elif d is None:
                return -1  # Replace None with 0
            else:
                return d

        def process_category_tag(df):
            df['category_tag'] = df['category_tag'].apply(replace_none_in_nested_dict)
            return df

        # Use this function before logging to wandb
        output = process_category_tag(output)
        columns = ["prompt", "response_a", "response_b", "category_tag"] if not args.vision else ["prompt", "image", "response_a", "response_b", "category_tag"]
        if args.vision:
            # read image_path into wandb Image
            output["image"] = output.image_path.map(lambda x: wandb.Image(x))

        wandb.log({"categories": wandb.Table(dataframe=output[columns])})

    if config["convert_to_json"] and os.path.isfile(config["output_file"]):
        # merge two data frames, but only take the fields from the cache data to overwrite the input data
        merge_columns = [category.name_tag for category in categories]
        print(f"Columns to be merged:\n{merge_columns}")

        input_data["uid"] = input_data.question_id.map(str) + input_data.tstamp.map(str)
        assert len(input_data) == len(input_data.uid.unique())

        # fastest way to merge
        assert os.path.isfile(config["output_file"])
        print("reading output file...")
        temp = pd.read_json(config["output_file"], lines=True)
        temp["uid"] = temp.question_id.map(str) + temp.tstamp.map(str)
        assert len(temp) == len(temp.uid.unique())

        assert "category_tag" in temp.columns
        output_dict = temp[["uid", "category_tag"]].set_index("uid")
        print("finalizing output_dict (should take less than 30 sec)")
        OUTPUT_DICT = output_dict.to_dict("index")

        print("begin merging (should take around 1 minute or less on large dataset)")
        input_data["category_tag"] = input_data.apply(category_merge, axis=1)
        print("merge completed")

        final_data = input_data.drop(
            columns=["prompt", "uid", "required_tasks"], errors="ignore"
        )
        final_data.to_json(
            config["output_file"][:-1], orient="records", indent=4, force_ascii=False
        )
