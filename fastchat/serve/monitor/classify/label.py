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

from category import Category


LOCK = threading.RLock()

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
):  
    for category in categories:
        conv = category.pre_process(question["prompt"])
        output = chat_completion_openai(
            model=model_name,
            messages=conv,
            temperature=temperature,
            max_tokens=max_tokens,
            api_dict=api_dict,
        )
        # Dump answers
        question[category.name_tag] = category.post_process(output)
    
    question.drop(["prompt", "uid", "task_required"], inplace=True)

    with LOCK:
        with open(answer_file, "a") as fout:
            fout.write(json.dumps(question.to_dict()) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    enter = input("Make sure your config file is properly configured. Press enter to continue.")
    if not enter == '':
        exit()
    
    config = make_config(args.config)
    
    API_MAX_RETRY = config["max_retry"]
    API_RETRY_SLEEP = config["retry_sleep"]
    API_ERROR_OUTPUT = config["error_output"]
    
    categories = [Category.create_category(name) for name in config["task_name"]]
    print(f"Following categories will be labeled:\n{[category.name_tag for category in categories]}")

    print("loading input data (might take min)")
    with open(config["input_file"], "rb") as f:
        data = orjson.loads(f.read())
    input_data = pd.DataFrame(data)
    not_labeled = input_data.copy()
    
    not_labeled["uid"] = not_labeled.apply(lambda row: row["question_id"] + str(row["tstamp"]), axis=1)
    not_labeled["task_required"] = [[category.name_tag for category in categories]] * len(not_labeled)
    print(f"{len(not_labeled)}# of input data just loaded")

    if config["cache_file"]:
        print("loading cache data")
        with open(config["cache_file"], "rb") as f:
            data = orjson.loads(f.read())
        cache_data = pd.DataFrame(data)
        cache_data["uid"] = cache_data.apply(lambda row: row["question_id"] + str(row["tstamp"]), axis=1)
        print(f"{len(cache_data)}# of cache data just loaded")

        for category in categories:
            if category.name_tag not in cache_data.columns:
                continue
            
            ids = cache_data[["uid", category.name_tag]].dropna(axis=0).uid.tolist()
            assert len(ids) == len(set(ids)), "qid + tstamp somehow not unique"
            ids = set(ids)
            print(f"found {len(ids)} # of existing {category.name_tag} data in cache file.")
            not_labeled["task_required"] = not_labeled.apply(lambda row: [name for name in row["task_required"] if name != category.name_tag] if row["uid"] in ids else row["task_required"], axis=1)

    if os.path.isfile(config["output_file"]):
        print("loading existing output")
        output_data = pd.read_json(config["output_file"], lines=True)
        output_data["uid"] = output_data.apply(lambda row: row["question_id"] + str(row["tstamp"]), axis=1)
        print(f"{len(output_data)}# of existing output just loaded")

        for category in categories:
            if category.name_tag not in output_data.columns:
                continue
            
            ids = output_data[["uid", category.name_tag]].dropna(axis=0).uid.tolist()
            assert len(ids) == len(set(ids)), "qid + tstamp somehow not unique"
            ids = set(ids)
            print(f"found {len(ids)} # of existing {category.name_tag} data in output file.")
            not_labeled["task_required"] = not_labeled.apply(lambda row: [name for name in row["task_required"] if name != category.name_tag] if row["uid"] in ids else row["task_required"], axis=1)

    not_labeled = not_labeled[not_labeled.task_required.map(lambda x: len(x) > 0)].copy()
    
    print(f"{len(not_labeled)} needs to be labeled")

    not_labeled["prompt"] = not_labeled.conversation_a.map(
        lambda convo: "\n".join([convo[i]["content"] for i in range(0, len(convo), 2)])
    )
    not_labeled["prompt"] = not_labeled.prompt.map(lambda x: x[:12500])
 
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
                [category for category in categories if category.name_tag in row["task_required"]],
            )
            futures.append(future)
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()

    if config["convert_to_json"]:
        # merge two data frames, but only take the fields from the cache data to overwrite the input data
        merge_columns = [category.name_tag for category in categories]
        print(f"Columns to be merged:\n{merge_columns}")
        
        input_data["uid"] = input_data.apply(lambda row: row["question_id"] + str(row["tstamp"]), axis=1)
        
        if config["cache_file"]:
            input_data = pd.merge(
                input_data, cache_data[["uid"] + [name for name in merge_columns if name in cache_data.columns]], on="uid", how="left"
            )
        input_data[[name for name in merge_columns if name not in input_data.columns]] = None

        if os.path.isfile(config["output_file"]):
            temp = pd.read_json(config["output_file"], lines=True)
            temp["uid"] = temp.apply(lambda row: row["question_id"] + str(row["tstamp"]), axis=1)
            input_data = pd.merge(
                input_data, temp[["uid"] + [name for name in merge_columns if name in temp.columns]], on="uid", how="left"
            )
            
            for category in categories:
                if category.name_tag + "_x" not in input_data.columns:
                    continue
                input_data[category.name_tag] = input_data[category.name_tag + "_x"].combine_first(input_data[category.name_tag + "_y"])
                input_data.drop(columns=[category.name_tag + "_x", category.name_tag + "_y"], inplace=True)
                print("data filled")

        final_data = input_data.drop(columns=["prompt"], errors="ignore")
        final_data.to_json(
            config["output_file"][:-1], orient="records", indent=4, force_ascii=False
        )