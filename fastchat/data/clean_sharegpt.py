"""
- Convert html to markdown with basic data cleaning.
- Deduplication.

Usage:
python3 -m fastchat.data.clean_sharegpt --in sharegpt_html.json --out sharegpt_clean.json
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import re
from typing import Dict, Union

import bs4
import markdownify  # == 0.11.6
from tqdm import tqdm


div_pattern = re.compile("<div.*?>")
span_pattern = re.compile("<span.*?>")
code_lang_pattern = re.compile(
    "```\s*" + "(.*?)" + "(?:Copy code)+" + "(.+?)" + "\s*?```", re.DOTALL
)
code_lang_format = "```\g<1>\n\g<2>\n```"
regenerate_pattern = re.compile("\d+ / \d+")
copy_chars_pattern = re.compile("Copy\d+ chars / \d+ words")
copy_code_pattern = re.compile("```(.*?)Copy code\s*```")


def reformat_code(val: str) -> str:
    # Input code format is:
    # ```
    # $<language>Copy code$<exact_code_here>
    #
    # ```
    # This function convert it into the correct markdown format
    return re.sub(code_lang_pattern, code_lang_format, val)


def html_to_markdown(val: str) -> str:
    # Remove all <div>. This is required to make intent work in code blocks.
    val = re.sub(div_pattern, "", val)
    # Remove all <span>. This is required to make underscores work in code blocks.
    val = re.sub(span_pattern, "", val)
    # Markdown to html
    val = markdownify.markdownify(val).strip()
    # Reformat code
    val = reformat_code(val)

    # Remove noisy "[number] / [number]" at the beginning
    noise = re.search(regenerate_pattern, val)
    if noise and noise.start() == 0:
        val = val[noise.end() :]
    # Remove noisy "Copy[number] chars / [number] words"
    val = re.sub(copy_chars_pattern, "", val)
    # Remove empty code block ```\nCopy code\n```
    val = re.sub(copy_code_pattern, "", val)

    # Strip
    val = val.replace("\n\n\n", "\n").strip()

    return val


def contain_blocked_words(val: str) -> bool:
    blocked_words = ["openai", "chatgpt"]
    for w in blocked_words:
        if w in val.lower():
            return True
    return False


def clean_html_one_sample(sample):
    roles = ["human", "gpt"]

    if len(sample["conversations"]) <= 1:
        return (sample, 1)

    # Adjust the offset for cases like https://sharegpt.com/c/VyaZlh4
    if sample["conversations"][0]["from"] != "human":
        sample["conversations"] = sample["conversations"][1:]
    if len(sample["conversations"]) <= 1:
        return (sample, 1)

    if sample["conversations"][-1]["from"] == "human":
        sample["conversations"] = sample["conversations"][:-1]
    if len(sample["conversations"]) <= 1:
        return (sample, 1)

    for i, c in enumerate(sample["conversations"]):
        if c["from"] != roles[i % 2]:
            return (sample, 2)

        if contain_blocked_words(c["value"]):
            return (sample, 3)

        try:
            new_val = html_to_markdown(c["value"])
        except (bs4.builder.ParserRejectedMarkup, AssertionError):
            return (sample, 4)

        c["value"] = new_val

    return (sample, 0)


def clean_html_all(content, begin, end):
    """
    Clean the source html files.
    """
    cnt_skip = 0
    cnt_blocked_words = 0
    cnt_wrong_format = 0
    cnt_parser_error = 0
    cnt_too_short = 0
    cnt_id_duplication = 0
    cnt_value_duplication = 0
    cnt_tag = 0

    content = content[begin:end]
    processed = []
    with ProcessPoolExecutor() as executor:
        for result in tqdm(
            executor.map(clean_html_one_sample, content), total=len(content)
        ):
            processed.append(result)

    visited = {}
    new_content = []
    for sample, error_code in tqdm(processed):
        cid = sample["id"]
        skipped = True

        if error_code != 0:
            if error_code == 1:
                print(f"id {cid} is too short")
                cnt_too_short += 1
            elif error_code == 2:
                print(f"id {cid} has a wrong format")
                cnt_wrong_format += 1
            elif error_code == 3:
                print(f"id {cid} contains blocked words")
                cnt_blocked_words += 1
            elif error_code == 4:
                print(f"id {cid} contains parser errors")
                cnt_parser_error += 1
            else:
                raise ValueError(f"Invalid error_code: {error_code}")
        elif cid in visited:
            print(f"id {cid} is an id duplication of {visited[cid]}")
            cnt_id_duplication += 1
        elif (
            sample["conversations"][1]["value"],
            len(sample["conversations"]),
        ) in visited:
            key = (sample["conversations"][1]["value"], len(sample["conversations"]))
            print(f"id {cid} is a value duplication of {visited[key]}")
            cnt_value_duplication += 1
        else:
            key = (sample["conversations"][1]["value"], len(sample["conversations"]))
            visited[cid] = visited[key] = cid
            skipped = False

        if not skipped:
            new_content.append(sample)
        else:
            cnt_skip += 1

    print(
        f"total: {len(content)}, skip: {cnt_skip}, new: {len(new_content)}, "
        f"cnt_blocked_words: {cnt_blocked_words}, cnt_parser_error: {cnt_parser_error}, "
        f"cnt_wrong_format: {cnt_wrong_format}, "
        f"cnt_too_short: {cnt_too_short}, cnt_id_duplication: {cnt_id_duplication}, "
        f"cnt_value_duplication: {cnt_value_duplication}, "
    )

    return new_content


def main(args):
    content = json.load(open(args["in_file"], "r"))
    content = clean_html_all(content, args["begin"], args["end"])
    json.dump(content, open(args["out_file"], "w"), indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, default="sharegpt_clean.json")
    parser.add_argument("--begin", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(vars(args))
