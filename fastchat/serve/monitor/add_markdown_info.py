import pandas as pd
import re
import markdown
import argparse

from tqdm import tqdm

tqdm.pandas()


def count_markdown_elements(markdown_text):
    # Initialize counters
    headers = 0
    lists = 0
    bold = 0

    # Count headers (# to ######)
    headers += len(re.findall(r"^#{1,6}\s", markdown_text, re.MULTILINE))

    # Count unordered list items
    lists += len(re.findall(r"^\s*[-*+]\s", markdown_text, re.MULTILINE))

    # Count ordered list items
    lists += len(re.findall(r"^\s*\d+\.\s", markdown_text, re.MULTILINE))

    # Count bold elements (both ** and __ syntax)
    bold += len(re.findall(r"\*\*[^*\n]+\*\*", markdown_text))
    bold += len(re.findall(r"__[^_\n]+__", markdown_text))

    return bold, lists, headers


def remove_code(answer, pattern):
    blocks = pattern.findall(answer)
    for block in blocks:
        answer = answer.replace(block, "")
    return answer


# def get_element_counts(df, column="conversation_a"):
#     pattern = re.compile("```([^`]*)```")
#     md = markdown.Markdown()

#     answers = df[column].map(lambda convo: '\n'.join([turn["content"] for turn in convo if turn["role"] == "assistant"]))
#     answers = answers.map(lambda answer: remove_code(answer, pattern))

#     results = []
#     for answer in tqdm(answers):
#         try:
#             results.append(count_markdown_elements(answer))
#         except Exception as e:
#             print(e)
#             results.append((-1, -1, -1))

#     return results


def get_element_counts(df, column="conversation_a"):
    pattern = re.compile("```([^`]*)```")
    answers = df[column].map(
        lambda convo: "\n".join(
            [turn["content"] for turn in convo if turn["role"] == "assistant"]
        )
    )
    results = answers.progress_map(
        lambda answer: count_markdown_elements(remove_code(answer, pattern))
    )

    return results.tolist()


def add_markdown_meta(row):
    conv_meta = {k: v for k, v in row["conv_metadata"].items()}
    conv_meta["sum_bold_count_a"] = row["markdown_meta_a"][0]
    conv_meta["sum_list_count_a"] = row["markdown_meta_a"][1]
    conv_meta["sum_header_count_a"] = row["markdown_meta_a"][2]

    conv_meta["sum_bold_count_b"] = row["markdown_meta_b"][0]
    conv_meta["sum_list_count_b"] = row["markdown_meta_b"][1]
    conv_meta["sum_header_count_b"] = row["markdown_meta_b"][2]

    return conv_meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    args = parser.parse_args()

    print("loading file...")
    data = pd.read_json(args.input_file)

    assert "conv_metadata" in data.columns

    temp = data[["question_id", "conv_metadata"]].copy()

    print("Processing conversation_a")
    temp["markdown_meta_a"] = get_element_counts(data, column="conversation_a")

    print("Processing conversation_b")
    temp["markdown_meta_b"] = get_element_counts(data, column="conversation_b")

    print("Post-processing...")
    data["conv_metadata"] = temp.apply(add_markdown_meta, axis=1)

    print("Saving to file...")
    data.to_json(args.output_file, orient="records", indent=4, force_ascii=False)
