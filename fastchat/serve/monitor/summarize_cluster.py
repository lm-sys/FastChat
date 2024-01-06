"""
Usage:
python3 summarize_cluster.py --in results_c20_kmeans_cluster.pkl --model gpt-4 --num-prompts 100
python3 summarize_cluster.py --in results_c20_kmeans_cluster.pkl --model azure-gpt-4-32k --num-prompts 200
"""
import argparse
import pickle

from fastchat.llm_judge.common import (
    chat_completion_openai,
    chat_completion_openai_azure,
    chat_completion_anthropic,
)
from fastchat.conversation import get_conv_template


def truncate_string(s, l):
    half = int(l // 2)
    return s[:half] + s[-half:] if len(s) > l else s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--num-prompts", type=int, default=100)
    args = parser.parse_args()

    model = args.model

    cluster_infos = pickle.load(open(args.input_file, "rb"))
    num_total_prompts = sum([x[0] for x in cluster_infos])

    topics = []
    percentages = []
    for i, info in enumerate(cluster_infos):
        num_samples, topk_prompts, random_prompts = info
        percentage = num_samples / num_total_prompts
        print(
            f"cluster {i}, #prompts {num_samples}, percentage: {percentage * 100:.2f}%"
        )
        instruct = "Given a list of user messages, use less than 8 words to summarize a central topic for all messages in English. Your output should only include a single line. Try to be specific."
        split = int(args.num_prompts * 0.8)
        prompt = "\n".join(
            [truncate_string(x, l=200) for x in topk_prompts[:split]]
            + [
                truncate_string(x, l=200)
                for x in random_prompts[: args.num_prompts - split]
            ]
        )
        prompt = "BEGIN OF THE MESSAGE LIST\n" + prompt + "\nEND OF THE MESSAGE LIST."

        if "azure-" in model:
            template_name = "chatgpt"
            completion_func = chat_completion_openai_azure
        elif "gpt" in model:
            template_name = "chatgpt"
            completion_func = chat_completion_openai
        elif "claude" in model:
            template_name = "claude"
            completion_func = chat_completion_anthropic

        conv = get_conv_template(template_name)
        conv.set_system_message(instruct)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)

        topic = completion_func(model, conv, temperature=0, max_tokens=256)
        print(topic)

        topics.append(topic)
        percentages.append(round(percentage, 6))

    print()
    print(f"topics: {topics}")
    print(f"percentages: {percentages}")
