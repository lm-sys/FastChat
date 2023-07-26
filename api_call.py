import re
import numpy
import sys
import os

FUNCTION_CALLING_PATH = ""  # provide path for the function to use
sys.path.append(FUNCTION_CALLING_PATH)
from contact import contact  # contact is the function to be used


def extract_features(result: str):
    function = re.search(r"(.*?)\nObservation:", result).group(1)

    function_name = re.search(r":\s*(\w+)", function)
    function_name = function_name.group(1) if function_name else None

    keyword = re.search(r'keyword="([^"]+)"', function)
    keyword = keyword.group(1) if keyword else None

    output = ""
    if function_name == "contact":
        try:
            database_result = contact(keyword=keyword)
            output += database_result
        except Exception as e:
            output += "Employee name doesn't exist or not mentioned"

    return output
