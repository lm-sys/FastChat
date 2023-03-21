import argparse
import json
import logging
import re
from typing import Dict, Union

import tqdm

logger = logging.getLogger(__name__)


def _get_html_tags(file_path: str):
    # Generate the list of html tags occured in the file.
    s = set()
    for l in open("file_path", "r"):
        for m in re.findall("</[^<>]+>", l):
            s.add(m)
    return s


def _reformat_all_code(val: str) -> str:
    # Input code format is:
    # ```
    # $<language>Copy code`$<exact_code_here>`
    # ```
    # This function convert it into the correct markdown format
    match_pattern = re.compile("```\n" + "([^`]+)" + "Copy code`" + "([^`]+)" +
                               "`\n```")
    repl_format = r"```\g<1>\n\g<2>\n```"
    return re.sub(match_pattern, repl_format, val)


def _html_to_markdown(val: str) -> str:
    """can handle enum, table and code. Code not in the best format."""
    import markdownify
    out = markdownify.markdownify(val)
    return _reformat_all_code(out)


def clean_html_source(content: Union[list, Dict], check_tag="", check_num=1):
    """
    clean the input json content.
    Args:
        content(Union[list, Dict]): json file loaded in memory.
        check_tag: a debug purpose arg. If a conversation contains the tag, log
          it before and after cleaning.
        check_num: number of matched conversations logged.
    """
    if len(check_tag) == 0:
        check_tag = None
    else:
        tag_cnt = 0
    BARRIER = "=" * 20 + "\n"

    for l in tqdm.tqdm(content):
        for c in l["conversations"]:
            try:
                new_val = _html_to_markdown(c["value"])
            except:
                logger.warning(BARRIER + c["value"] + BARRIER +
                               "The above value is kept unchanged.")
                new_val = c["value"]
            if (check_tag is not None and check_tag in c["value"]
                    and tag_cnt < check_num):
                logger.debug(BARRIER + c["value"] + "\n" + BARRIER + new_val +
                             "\n" + BARRIER + "\n")
                tag_cnt += 1
                if tag_cnt == check_num:
                    break
            c["value"] = new_val
    return content


def main(args):
    content = json.load(open(args.file_path, "r"))
    content = clean_html_source(content, args.check_tag, args.check_num)
    json.dump(content, open(args.output_path, "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type=str)
    parser.add_argument('--output-path', type=str, default="./cleaned.json")
    parser.add_argument('--check-tag', type=str, default="")
    parser.add_argument('--check-num', type=int, default=1)
    args = parser.parse_args()
    main(args)
