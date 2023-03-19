import json
import logging

import tqdm

logger = logging.getLogger(__name__)

RAW_CONVERSATION_PATH = "./all_conversations.json"
OUTPUT_PATH = "./cleaned.json"


def _get_html_tags(file_path: str):
    # Generate the to_handle_list. So long and cannot be handle one by one.
    import re
    s = set()
    for l in open("file_path", "r"):
        for m in re.findall("</[^<>]+>", l):
            s.add(m)
    return s


def _html_to_plain_text(val: str) -> str:
    from html.parser import HTMLParser

    class HtmlFilter(HTMLParser):
        text = ""

        def handle_data(self, data):
            self.text += data + "\n"

    f = HtmlFilter()
    f.feed(val)
    return f.text[:-1]


def _html_to_rich_text(val: str) -> str:
    """This one can handle enum, but cannot handle code."""
    import html2text
    h = html2text.HTML2Text()
    return h.handle(val)


def _html_to_markdown(val: str) -> str:
    """can handle enum and code. Table not tested. Code not in best format."""
    import markdownify
    out = markdownify.markdownify(val)
    # TODO: the output format is:
    # ```
    # pythonCode`$<exact_code_here>`
    # ```
    # We need to convert it into the correct markdown format
    # TODO: test whether table is correctly converted
    return out


# Handles li correctly
json_file = json.load(open(RAW_CONVERSATION_PATH, "r"))
for l in tqdm.tqdm(json_file):
    for c in l["conversations"]:
        try:
            new_val = _html_to_markdown(c["value"])
        except:
            break_line = "=" * 20 + "\n"
            logger.warning(
                break_line + c["value"] + break_line +
                "The above value is not correctly handled, keep it unchanged.")
            new_val = c["value"]
        c["value"] = new_val
json.dump(json_file, open(OUTPUT_PATH, "w"))
