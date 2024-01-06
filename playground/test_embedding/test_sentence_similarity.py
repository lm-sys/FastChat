import json
import os

import numpy as np
import openai
import requests
from scipy.spatial.distance import cosine


def get_embedding_from_api(word, model="vicuna-7b-v1.5"):
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


def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)


def print_cosine_similarity(embeddings, texts):
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = cosine_similarity(embeddings[texts[i]], embeddings[texts[j]])
            print(f"Cosine similarity between '{texts[i]}' and '{texts[j]}': {sim:.2f}")


texts = [
    "The quick brown fox",
    "The quick brown dog",
    "The fast brown fox",
    "A completely different sentence",
]

embeddings = {}
for text in texts:
    embeddings[text] = get_embedding_from_api(text)

print("Vicuna-7B:")
print_cosine_similarity(embeddings, texts)

for text in texts:
    embeddings[text] = get_embedding_from_api(text, model="text-similarity-ada-001")

print("text-similarity-ada-001:")
print_cosine_similarity(embeddings, texts)

for text in texts:
    embeddings[text] = get_embedding_from_api(text, model="text-embedding-ada-002")

print("text-embedding-ada-002:")
print_cosine_similarity(embeddings, texts)
