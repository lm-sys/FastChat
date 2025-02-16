from typing import Dict, List
import json
import requests
import math
import os.path
import pandas as pd


P2L_BASE_URL = None
P2L_API_KEY = None


def query_p2l_endpoint(
    prompt: list[str], base_url: str, api_key: str
) -> Dict[str, List]:
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }

    payload = {"prompt": prompt}

    try:
        response = requests.post(
            f"{base_url}/predict", headers=headers, data=json.dumps(payload)
        )
        response.raise_for_status()
        result = response.json()
        return result

    except Exception as err:
        raise err


def get_p2l_endpoint_models(base_url: str, api_key: str) -> List[str]:
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }

    try:
        response = requests.get(f"{base_url}/models", headers=headers)
        response.raise_for_status()
        result = response.json()
        return result["models"]

    except Exception as err:
        print(f"An error occurred: {err}")


def scale_and_offset(
    ratings,
    models,
    scale=400,
    init_rating=1000,
    baseline_model="Mixtral-8x7B-Instruct-v0.1",
    baseline_rating=1114,
):
    scaled_ratings = (ratings * scale / math.log(10)) + init_rating
    if baseline_model in models:
        scaled_ratings += baseline_rating - scaled_ratings[baseline_model]
    return scaled_ratings


def get_p2l_leaderboard(
    prompt: str,
) -> Dict[str, float]:
    # cache the p2l model_list
    if os.path.isfile("p2l_model_list.json"):
        with open("p2l_model_list.json", "r") as file:
            model_list = json.load(file)
    else:
        model_list = get_p2l_endpoint_models(P2L_BASE_URL, P2L_API_KEY)
        with open("p2l_model_list.json", "w") as file:
            json.dump(model_list, file)

    lb = query_p2l_endpoint([prompt], P2L_BASE_URL, P2L_API_KEY)

    lb = pd.Series(data=lb["coefs"], index=model_list)

    lb_scaled = scale_and_offset(lb, model_list).sort_values(ascending=False)

    return lb_scaled
