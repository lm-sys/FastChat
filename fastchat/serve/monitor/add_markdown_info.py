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
    return conv_meta | row["markdown_meta_a"] | row["markdown_meta_b"]


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
