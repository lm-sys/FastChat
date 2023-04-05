"""
Convert html to markdown with basic data cleaning.

Usage:
python3 -m fastchat.data.clean_sharegpt --in sharegpt_html.json --out sharegpt_clean.json
"""
import argparse
import json
import logging
import re
from typing import Dict, Union

import bs4
import markdownify  # == 0.11.6
import tqdm


div_pattern = re.compile("<div.*?>")
span_pattern = re.compile("<span.*?>")
code_lang_pattern = re.compile("```\s*" + "(.*?)" + "(?:Copy code)+" + "(.+?)" + "\s*?```", re.DOTALL)
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
        val = val[noise.end():]
    # Remove noisy "Copy[number] chars / [number] words"
    val = re.sub(copy_chars_pattern, "", val)
    # Remove empty code block ```\nCopy code\n```
    val = re.sub(copy_code_pattern, "", val)

    # Strip
    val = val.replace("\n\n\n", "\n").strip()

    if args.debug:
        print(val)
        exit()

    return val


def should_skip(val: str) -> bool:
    black_list = ["openai", "chatgpt"]
    for w in black_list:
        if w in val.lower():
            return True
    return False


def clean_html_source(content, begin, end, check_tag, check_num):
    """
    clean the input json content.
    Args:
        content: json file loaded in memory.
        check_tag: a debug purpose arg. If a conversation contains the tag, log
          it before and after cleaning.
        check_num: number of matched conversations logged.
    """
    BARRIER = "\n" + "=" * 20 + "\n"
    skip_cnt = 0
    tag_cnt = 0

    content = content[begin:end]
    new_content = []

    for sample in tqdm.tqdm(content):
        skipped = False

        if len(sample["conversations"]) <= 1:
            # The conversation is too short
            skipped = True
        else:
            for c in sample["conversations"]:
                if should_skip(c["value"]):
                    skipped = True
                    break

                try:
                    new_val = html_to_markdown(c["value"])
                except (bs4.builder.ParserRejectedMarkup, AssertionError):
                    skipped = True
                    break

                c["value"] = new_val

                # Debug
                if (check_tag is not None and check_tag in c["value"]
                        and tag_cnt < check_num):
                    logging.debug(BARRIER + c["value"] + "\n" + BARRIER + new_val +
                                  "\n" + BARRIER + "\n")
                    tag_cnt += 1
                    if tag_cnt == check_num:
                        break

        if not skipped:
            new_content.append(sample)
        else:
            skip_cnt += 1

    print(f"total: {len(content)}, skip: {skip_cnt}, new: {len(new_content)}")
    return new_content


def main(args):
    content = json.load(open(args['in_file'], "r"))
    content = clean_html_source(
        content, args['begin'], args['end'],
        args['check_tag'], args['check_num'])
    json.dump(content, open(args['out_file'], "w"), indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, default="sharegpt_clean.json")
    parser.add_argument("--begin", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--check-tag", type=str)
    parser.add_argument("--check-num", type=int, default=1)
    args = parser.parse_args()
    main(vars(args))
