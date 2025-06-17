"""
Usage:
python3 -m playground.benchmark.benchmark_api_provider --api-endpoint-file api_endpoints.json --output-file ./benchmark_results.json --random-questions metadata_sampled.json
"""
import argparse
import json
import time

import numpy as np

from fastchat.serve.api_provider import get_api_provider_stream_iter
from fastchat.serve.gradio_web_server import State
from fastchat.serve.vision.image import Image


class Metrics:
    def __init__(self):
        self.ttft = None
        self.avg_token_time = None

    def to_dict(self):
        return {"ttft": self.ttft, "avg_token_time": self.avg_token_time}


def sample_image_and_question(random_questions_dict, index):
    # message = np.random.choice(random_questions_dict)
    message = random_questions_dict[index]
    question = message["question"]
    path = message["path"]

    if isinstance(question, list):
        question = question[0]

    return (question, path)


def call_model(
    conv,
    model_name,
    model_api_dict,
    state,
    temperature=0.4,
    top_p=0.9,
    max_new_tokens=2048,
):
    prev_message = ""
    prev_time = time.time()
    CHARACTERS_PER_TOKEN = 4
    metrics = Metrics()

    stream_iter = get_api_provider_stream_iter(
        conv, model_name, model_api_dict, temperature, top_p, max_new_tokens, state
    )
    call_time = time.time()
    token_times = []
    for i, data in enumerate(stream_iter):
        output = data["text"].strip()
        if i == 0:
            metrics.ttft = time.time() - call_time
            prev_message = output
            prev_time = time.time()
        else:
            token_diff_length = (len(output) - len(prev_message)) / CHARACTERS_PER_TOKEN
            if token_diff_length == 0:
                continue

            token_diff_time = time.time() - prev_time
            token_time = token_diff_time / token_diff_length
            token_times.append(token_time)
            prev_time = time.time()

    metrics.avg_token_time = np.mean(token_times)
    return metrics


def run_benchmark(model_name, model_api_dict, random_questions_dict, num_calls=20):
    model_results = []

    for index in range(num_calls):
        state = State(model_name)
        text, image_path = sample_image_and_question(random_questions_dict, index)
        max_image_size_mb = 5 / 1.5

        images = [
            Image(url=image_path).to_conversation_format(
                max_image_size_mb=max_image_size_mb
            )
        ]
        message = (text, images)

        state.conv.append_message(state.conv.roles[0], message)
        state.conv.append_message(state.conv.roles[1], None)

        metrics = call_model(state.conv, model_name, model_api_dict, state)
        model_results.append(metrics.to_dict())

    return model_results


def benchmark_models(api_endpoint_info, random_questions_dict, models):
    results = {model_name: [] for model_name in models}

    for model_name in models:
        model_results = run_benchmark(
            model_name,
            api_endpoint_info[model_name],
            random_questions_dict,
            num_calls=20,
        )
        results[model_name] = model_results

    print(results)
    return results


def main(api_endpoint_file, random_questions, output_file):
    api_endpoint_info = json.load(open(api_endpoint_file))
    random_questions_dict = json.load(open(random_questions))
    models = ["reka-core-20240501", "gpt-4o-2024-05-13"]

    models_results = benchmark_models(api_endpoint_info, random_questions_dict, models)

    with open(output_file, "w") as f:
        json.dump(models_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-endpoint-file", required=True)
    parser.add_argument("--random-questions", required=True)
    parser.add_argument("--output-file", required=True)

    args = parser.parse_args()

    main(args.api_endpoint_file, args.random_questions, args.output_file)
