import json
import os

import numpy as np
import openai
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


np.set_printoptions(threshold=10000)


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


def train_random_forest(df):
    X_train, X_test, y_train, y_test = train_test_split(
        list(df.embedding.values), df.Score, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    report = classification_report(y_test, preds)
    accuracy = accuracy_score(y_test, preds)
    return clf, accuracy, report


input_datapath = "amazon_fine_food_review.csv"
if not os.path.exists(input_datapath):
    raise Exception(
        f"Please download data from: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews"
    )

df = create_embedding_data_frame(input_datapath, "vicuna-7b-v1.1")
clf, accuracy, report = train_random_forest(df)
print(f"Vicuna-7b-v1.1 accuracy:{accuracy}")
df = create_embedding_data_frame(input_datapath, "text-similarity-ada-001")
clf, accuracy, report = train_random_forest(df)
print(f"text-similarity-ada-001 accuracy:{accuracy}")
df = create_embedding_data_frame(input_datapath, "text-embedding-ada-002")
clf, accuracy, report = train_random_forest(df)
print(f"text-embedding-ada-002 accuracy:{accuracy}")
