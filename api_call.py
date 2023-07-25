import re
import numpy
import sys
import os

sys.path.append("/home/minhvn/workspace/llm/data-generation")
from tools.contact import contact


def extract_features(result: str):
    function = re.search(r"(.*?)\nObservation:", result).group(1)

    function_name = re.search(r":\s*(\w+)", function)
    function_name = function_name.group(1) if function_name else None

    keyword = re.search(r'keyword="([^"]+)"', function)
    keyword = keyword.group(1) if keyword else None

    output = ""
    if function_name == "contact":
        output += contact(keyword=keyword)
    return output
