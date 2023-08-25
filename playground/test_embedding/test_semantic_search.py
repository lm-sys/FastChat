import json
import os

import numpy as np
import openai
import pandas as pd
import requests
from scipy.spatial.distance import cosine


def cosine_similarity(vec1, vec2):
    try:
        return 1 - cosine(vec1, vec2)
    except:
        print(vec1.shape, vec2.shape)


def get_embedding_from_api(word, model="vicuna-7b-v1.1"):
    if "ada" in model:
        resp = openai.Embedding.create(
            model=model,
            input=word,
        )
        embedding = np.array(resp["data"][0]["embedding"])
        return embedding

    url = "http://localhost:8000/v1/embeddings"
    headers = {"Content-Type": "application/json"}
    data = json.dumps({"model": model, "input": word})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        embedding = np.array(response.json()["data"][0]["embedding"])
        return embedding
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


def create_embedding_data_frame(data_path, model, max_tokens=500):
    df = pd.read_csv(data_path, index_col=0)
    df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
    df = df.dropna()
    df["combined"] = (
        "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
    )
    top_n = 1000
    df = df.sort_values("Time").tail(top_n * 2)
    df.drop("Time", axis=1, inplace=True)

    df["n_tokens"] = df.combined.apply(lambda x: len(x))
    df = df[df.n_tokens <= max_tokens].tail(top_n)
    df["embedding"] = df.combined.apply(lambda x: get_embedding_from_api(x, model))
    return df


def search_reviews(df, product_description, n=3, pprint=False, model="vicuna-7b-v1.1"):
    product_embedding = get_embedding_from_api(product_description, model=model)
    df["similarity"] = df.embedding.apply(
        lambda x: cosine_similarity(x, product_embedding)
    )

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .combined.str.replace("Title: ", "")
        .str.replace("; Content:", ": ")
    )
    if pprint:
        for r in results:
            print(r[:200])
            print()
    return results


def print_model_search(input_path, model):
    print(f"Model: {model}")
    df = create_embedding_data_frame(input_path, model)
    print("search: delicious beans")
    results = search_reviews(df, "delicious beans", n=5, model=model)
    print(results)
    print("search: whole wheat pasta")
    results = search_reviews(df, "whole wheat pasta", n=5, model=model)
    print(results)
    print("search: bad delivery")
    results = search_reviews(df, "bad delivery", n=5, model=model)
    print(results)


input_datapath = "amazon_fine_food_review.csv"
if not os.path.exists(input_datapath):
    raise Exception(
        f"Please download data from: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews"
    )


print_model_search(input_datapath, "vicuna-7b-v1.1")
print_model_search(input_datapath, "text-similarity-ada-001")
print_model_search(input_datapath, "text-embedding-ada-002")
