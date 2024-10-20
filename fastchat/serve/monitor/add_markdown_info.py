import pandas as pd
import re
import argparse

from tqdm import tqdm

tqdm.pandas()


def count_markdown_elements(markdown_text, suffix):
    counters = {
        f"header_count{suffix}": {
            "h1": len(re.findall(r"^#{1}\s", markdown_text, re.MULTILINE)),
            "h2": len(re.findall(r"^#{2}\s", markdown_text, re.MULTILINE)),
            "h3": len(re.findall(r"^#{3}\s", markdown_text, re.MULTILINE)),
            "h4": len(re.findall(r"^#{4}\s", markdown_text, re.MULTILINE)),
            "h5": len(re.findall(r"^#{5}\s", markdown_text, re.MULTILINE)),
            "h6": len(re.findall(r"^#{6}\s", markdown_text, re.MULTILINE)),
        },
        f"list_count{suffix}": {
            "ordered": len(re.findall(r"^\s*\d+\.\s", markdown_text, re.MULTILINE)),
            "unordered": len(re.findall(r"^\s*[-*+]\s", markdown_text, re.MULTILINE)),
        },
        f"bold_count{suffix}": {
            "**": len(re.findall(r"\*\*[^*\n]+\*\*", markdown_text)),
            "__": len(re.findall(r"__[^_\n]+__", markdown_text)),
        },
    }
    return counters


def remove_pattern(answer, pattern):
    blocks = pattern.findall(answer)
    for block in blocks:
        answer = answer.replace(block, "")
    return answer


def get_element_counts(df, column):
    pattern = re.compile("```([^`]*)```")
    answers = df[column].map(
        lambda convo: "\n".join(
            [turn["content"] for turn in convo if turn["role"] == "assistant"]
        )
    )
    results = answers.progress_map(
        lambda answer: count_markdown_elements(
            remove_pattern(answer, pattern),
            suffix=column[-2:],  # Remove code block first
        )
    )

    return results.tolist()


def add_markdown_meta(row):
    conv_meta = {k: v for k, v in row["conv_metadata"].items()}
    return conv_meta | row["markdown_meta_a"] | row["markdown_meta_b"] | {
            "friendliness_a": row["friendliness_a"],
            "friendliness_b": row["friendliness_b"],
        } | {
            "hesitant_a": row["hesitant_a"],
            "hesitant_b": row["hesitant_b"],
        }

from fastchat.serve.monitor.utils_llm import get_llm_output
# doing things here
def get_score(judgment, pattern):
    print(judgment) 
    matches = pattern.findall(judgment.replace("\n", "").lower())
    matches = [m for m in matches if m != ""]
    if len(set(matches)) == 0:
        return None
    elif len(set(matches)) == 1:
        return matches[0]
    else:
        return None
        

def add_friendliness(row, api_key):
    systems_prompt = "Given a prompt and responses from two LLMs (A and B), your task is to determine which response is more friendly. If both responses are equally friendly, respond with equal. Do not let the order of the responses influence your decision. Do NOT reply or expand on any of the responses, only reply with which response is more friendly.\n\nOutput your verdict in the following format:<decision>\n[A/B/equal]\n</decision>. Do NOT explain."
    prompt = f"Prompt: {row['prompt']}\n\nResponse A: {row['response_a']}\n\nResponse B: {row['response_b']}"
    response = get_llm_output("gpt-4o-mini", api_key, systems_prompt, prompt)
    pattern = response.replace("\n", "").lower().replace("<decision>", "").replace("</decision>", "").replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace("<", "").replace(">", "")
    if pattern == "a":
        return 1
    elif pattern == "b":
        return 0
    elif pattern == "equal":
        return 0.5
    else:
        print(f"Error: {pattern}")
        return 0.5

def add_hesitant(row, api_key):
    systems_prompt = "Given a prompt and responses from two LLMs (A and B), your task is to determine which response is more hesitant. Consider factors like adding disclaimers, stating when they are unsure, reccomending the user seek outside information instead, and refusing to give a complete answer. If both responses are equally hesitatnt, respond with equal. Do not let the order of the responses influence your decision. Do NOT reply or expand on any of the responses, only reply with which response is more hesitant.\n\nOutput your verdict in the following format:<decision>\n[A/B/equal]\n</decision>. Do NOT explain."
    prompt = f"Prompt: {row['prompt']}\n\nResponse A: {row['response_a']}\n\nResponse B: {row['response_b']}"
    response = get_llm_output("gpt-4o-mini", api_key, systems_prompt, prompt)
    pattern = response.replace("\n", "").lower().replace("<decision>", "").replace("</decision>", "").replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace("<", "").replace(">", "").strip()
    if "a" in pattern and "b" not in pattern and "equal" not in pattern:
        return 1
    elif "b" in pattern and "a" not in pattern and "equal" not in pattern:
        return 0
    elif pattern == "equal":
        return 0.5
    else:
        print(f"Error: {pattern}")
        return 0.5

from concurrent.futures import ThreadPoolExecutor, as_completed
def process_data_in_threads(dataframe, api_key, max_workers=64, function=add_friendliness):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(
            lambda row: function(row, api_key),
            [row for _, row in dataframe.iterrows()]
        ))
    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=False)
    parser.add_argument("--api_url", type=str, required=False)
    args = parser.parse_args()

    print("loading file...")
    data = pd.read_json(args.input_file)

    assert "conv_metadata" in data.columns

    temp = data[["question_id", "conv_metadata"]].copy()

    data["prompt"] = data.conversation_a.map(
            lambda convo: "\n".join([convo[i]["content"] for i in range(0, len(convo), 2)])
        )
    data["prompt"] = data.prompt.map(lambda x: x[:12500])
    get_content = lambda c: c if type(c) == str else c[0]
    data["response_a"] = data.conversation_a.map(
        lambda convo: "\n".join(
            [get_content(convo[i]["content"]) for i in range(1, len(convo), 2)]
        )
    )
    data["response_a"] = data.response_a.map(lambda x: x[:12500])
    data["response_b"] = data.conversation_b.map(
        lambda convo: "\n".join(
            [get_content(convo[i]["content"]) for i in range(1, len(convo), 2)]
        )
    )
    data["response_b"] = data.response_b.map(lambda x: x[:12500])

    print("Processing friendliness")
    temp["friendliness_a"] = process_data_in_threads(data, args.api_key)
    temp["friendliness_b"] = 1 - temp["friendliness_a"]

    print("Processing hesitant")
    temp["hesitant_a"] = process_data_in_threads(data, args.api_key, function=add_hesitant)
    temp["hesitant_b"] = 1 - temp["hesitant_a"]

    print("Processing conversation_a")
    temp["markdown_meta_a"] = get_element_counts(data, column="conversation_a")

    print("Processing conversation_b")
    temp["markdown_meta_b"] = get_element_counts(data, column="conversation_b")
    print(temp)

    print("Post-processing...")
    data["conv_metadata"] = temp.apply(add_markdown_meta, axis=1)
    print(data.iloc[0]['conv_metadata'])

    print("Saving to file...")
    data.to_json(args.output_file, orient="records", indent=4, force_ascii=False)
