import json
import logging
import re

import tqdm

logger = logging.getLogger(__name__)

RAW_CONVERSATION_PATH = "./all_conversations.json"
OUTPUT_PATH = "./cleaned.json"
CHECK_TAG = None    # to check the conversion of code, use /code.
CHECK_NUM = 1       # few examples are enough
BARRIER = "=" * 20 + "\n"

if CHECK_TAG is not None:
    tag_cnt = 0
    logger.setLevel(logging.DEBUG)


def _get_html_tags(file_path: str):
    # Generate the to_handle_list. So long and cannot be handle one by one.
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
    match_pattern = re.compile("```\n" + "([^`]+)" + "Copy code`" + "([^`]+)" + "`\n```")
    repl_format = r"```\g<1>\n\g<2>\n```"
    return re.sub(match_pattern, repl_format, val)


def _html_to_markdown(val: str) -> str:
    """can handle enum, table and code. Code not in the best format."""
    import markdownify
    out = markdownify.markdownify(val)
    return _reformat_all_code(out)


json_file = json.load(open(RAW_CONVERSATION_PATH, "r"))
for l in tqdm.tqdm(json_file):
    for c in l["conversations"]:
        try:
            new_val = _html_to_markdown(c["value"])
        except:
            logger.warning(
                BARRIER + c["value"] + BARRIER +
                "The above value is not correctly handled, keep it unchanged.")
            new_val = c["value"]
        if (CHECK_TAG is not None and CHECK_TAG in c["value"]
                and tag_cnt < CHECK_NUM):
            logger.debug(BARRIER + c["value"] + "\n" + BARRIER + new_val +
                         "\n" + BARRIER + "\n")
            tag_cnt += 1
            if tag_cnt == CHECK_NUM:
        c["value"] = new_val
json.dump(json_file, open(OUTPUT_PATH, "w"))
