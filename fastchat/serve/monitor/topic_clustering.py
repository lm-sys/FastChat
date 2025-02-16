"""

Usage:
python3 topic_clustering.py --in arena.json --english-only --min-length 32
python3 topic_clustering.py --in clean_conv_20230809_100k.json --english-only --min-length 32 --max-length 1536
"""
import argparse
import json
import pickle
import string
import time

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sklearn.cluster import KMeans, AgglomerativeClustering
import torch
from tqdm import tqdm
from openai import OpenAI

from fastchat.utils import detect_language


def remove_punctuation(input_string):
    # Make a translator object to remove all punctuation
    translator = str.maketrans("", "", string.punctuation)

    # Use the translator object to remove the punctuation
    no_punct = input_string.translate(translator)
    return no_punct


def read_texts(input_file, min_length, max_length, english_only):
    visited = set()
    texts = []

    lines = json.load(open(input_file, "r"))

    for l in tqdm(lines):
        if "text" in l:
            line_texts = [l["text"]]
        elif "conversation_a" in l:
            line_texts = [
                x["content"] for x in l["conversation_a"] if x["role"] == "user"
            ]
        elif "conversation" in l:
            line_texts = [
                x["content"] for x in l["conversation"] if x["role"] == "user"
            ]
        elif "turns" in l:
            line_texts = l["turns"]

        for text in line_texts:
            text = text.strip()

            # Filter language
            if english_only:
                lang = detect_language(text)
                if lang != "English":
                    continue

            # Filter short or long prompts
            if min_length:
                if len(text) < min_length:
                    continue

            if max_length:
                if len(text) > max_length:
                    continue

            # De-duplication
            words = sorted([x.lower() for x in remove_punctuation(text).split(" ")])
            words = "".join(words)
            if words in visited:
                continue

            visited.add(words)
            texts.append(text)
    return np.array(texts)


def get_embeddings(texts, model_name, batch_size):
    if model_name == "text-embedding-ada-002":
        client = OpenAI()
        texts = texts.tolist()

        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size)):
            text = texts[i : i + batch_size]
            responses = client.embeddings.create(input=text, model=model_name).data
            embeddings.extend([data.embedding for data in responses])
        embeddings = torch.tensor(embeddings)
    else:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            device="cuda",
            convert_to_tensor=True,
        )

    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu()


def run_k_means(embeddings, num_clusters):
    np.random.seed(42)
    clustering_model = KMeans(n_clusters=num_clusters, n_init="auto")
    clustering_model.fit(embeddings.numpy())
    centers = torch.from_numpy(clustering_model.cluster_centers_)
    labels = torch.from_numpy(clustering_model.labels_)

    # Sort labels
    classes, counts = np.unique(labels, return_counts=True)
    indices = np.argsort(counts)[::-1]
    classes = [classes[i] for i in indices]
    new_labels = torch.empty_like(labels)
    new_centers = torch.empty_like(centers)
    for i, c in enumerate(classes):
        new_labels[labels == c] = i
        new_centers[i] = centers[c]
    return new_centers, new_labels


def run_agg_cluster(embeddings, num_clusters):
    np.random.seed(42)
    clustering_model = AgglomerativeClustering(n_clusters=num_clusters)
    clustering_model.fit(embeddings)
    labels = torch.from_numpy(clustering_model.labels_)

    # Sort labels
    classes, counts = np.unique(labels, return_counts=True)
    indices = np.argsort(counts)[::-1]
    classes = [classes[i] for i in indices]
    new_labels = torch.empty_like(labels)
    for i, c in enumerate(classes):
        new_labels[labels == c] = i

    # Compute centers
    centers = []
    for i in range(len(classes)):
        centers.append(embeddings[new_labels == i].mean(axis=0, keepdim=True))
    centers = torch.cat(centers)
    return centers, new_labels


def run_hdbscan_cluster(embeddings):
    import hdbscan

    np.random.seed(42)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    labels = torch.from_numpy(clusterer.fit_predict(embeddings))

    # Sort labels
    classes, counts = np.unique(labels, return_counts=True)
    indices = np.argsort(counts)[::-1]
    classes = [classes[i] for i in indices]
    new_labels = torch.empty_like(labels)
    for i, c in enumerate(classes):
        new_labels[labels == c] = i

    # Compute centers
    centers = []
    for i in range(len(classes)):
        centers.append(embeddings[new_labels == i].mean(axis=0, keepdim=True))
    centers = torch.cat(centers)
    return centers, new_labels


def get_topk_indices(centers, labels, embeddings, topk):
    indices = []
    arange = torch.arange(len(labels))
    counts = torch.unique(labels, return_counts=True)[1]
    topk = min(topk, counts.min().item())
    for i in range(len(centers)):
        tmp_indices = labels == i
        tmp_arange = arange[tmp_indices]
        tmp_embeddings = embeddings[tmp_indices]

        scores = cos_sim(centers[i].unsqueeze(0), tmp_embeddings)[0]
        sorted_indices = torch.flip(torch.argsort(scores), dims=[0])
        indices.append(tmp_arange[sorted_indices[:topk]].unsqueeze(0))
    return torch.cat(indices)


def print_topk(texts, labels, topk_indices, show_cut_off):
    ret = ""
    for k in range(len(topk_indices)):
        num_samples = torch.sum(labels == k).item()

        ret += "=" * 20 + f" cluster {k}, #samples: {num_samples} " + "=" * 20 + "\n"
        for idx in topk_indices[k]:
            ret += "PROMPT: " + texts[idx][:show_cut_off] + "\n"
        ret += "=" * 40 + "\n\n"

    return ret


def get_cluster_info(texts, labels, topk_indices):
    np.random.seed(42)

    cluster_info = []
    for k in range(len(topk_indices)):
        num_samples = torch.sum(labels == k).item()
        topk_prompts = []
        for idx in topk_indices[k]:
            topk_prompts.append(texts[idx])
        random_prompts = []
        for idx in range(len(topk_indices)):
            random_prompts.append(np.random.choice(texts))
        cluster_info.append((num_samples, topk_prompts, random_prompts))

    return cluster_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--model", type=str, default="all-mpnet-base-v2")
    # default="all-MiniLM-L12-v2")
    # default="multi-qa-distilbert-cos-v1")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-length", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--english-only", action="store_true")
    parser.add_argument("--num-clusters", type=int, default=20)
    parser.add_argument(
        "--cluster-alg",
        type=str,
        choices=["kmeans", "aggcls", "HDBSCAN"],
        default="kmeans",
    )
    parser.add_argument("--show-top-k", type=int, default=200)
    parser.add_argument("--show-cut-off", type=int, default=512)
    parser.add_argument("--save-embeddings", action="store_true")
    parser.add_argument("--embeddings-file", type=str, default=None)
    args = parser.parse_args()

    num_clusters = args.num_clusters
    show_top_k = args.show_top_k
    show_cut_off = args.show_cut_off

    texts = read_texts(
        args.input_file, args.min_length, args.max_length, args.english_only
    )
    print(f"#text: {len(texts)}")

    if args.embeddings_file is None:
        embeddings = get_embeddings(texts, args.model, args.batch_size)
        if args.save_embeddings:
            # allow saving embedding to save time and money
            torch.save(embeddings, "embeddings.pt")
    else:
        embeddings = torch.load(args.embeddings_file)
    print(f"embeddings shape: {embeddings.shape}")

    if args.cluster_alg == "kmeans":
        centers, labels = run_k_means(embeddings, num_clusters)
    elif args.cluster_alg == "aggcls":
        centers, labels = run_agg_cluster(embeddings, num_clusters)
    elif args.cluster_alg == "HDBSCAN":
        centers, labels = run_hdbscan_cluster(embeddings)
    else:
        raise ValueError(f"Invalid clustering algorithm: {args.cluster_alg}")

    topk_indices = get_topk_indices(centers, labels, embeddings, args.show_top_k)
    topk_str = print_topk(texts, labels, topk_indices, args.show_cut_off)
    num_clusters = len(centers)

    # Dump results
    filename_prefix = f"results_c{num_clusters}_{args.cluster_alg}"
    print(topk_str)
    with open(filename_prefix + "_topk.txt", "w") as fout:
        fout.write(topk_str)

    with open(filename_prefix + "_all.jsonl", "w") as fout:
        for i in range(len(centers)):
            tmp_indices = labels == i
            tmp_embeddings = embeddings[tmp_indices]
            tmp_texts = texts[tmp_indices]

            scores = cos_sim(centers[i].unsqueeze(0), tmp_embeddings)[0]
            sorted_indices = torch.flip(torch.argsort(scores), dims=[0])

            for text, score in zip(tmp_texts[sorted_indices], scores[sorted_indices]):
                obj = {"cluster": i, "text": text, "sim": score.item()}
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    cluster_info = get_cluster_info(texts, labels, topk_indices)
    with open(filename_prefix + "_cluster.pkl", "wb") as fout:
        pickle.dump(cluster_info, fout)
