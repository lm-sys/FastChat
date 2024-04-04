import re
import json
import argparse
import multiprocessing as mp

import nltk
from tqdm import tqdm
from nltk.tokenize import word_tokenize


def is_code_conversation(text: str) -> tuple[bool, list[str]]:
    """Check if the text is a code conversation"""

    if "```plaintext" in text:
        lines = text.split("\n")
        line1_idx = [idx for idx, line in enumerate(lines) if "```plaintext" in line][0]
        line2_idx = [
            line1_idx + 1 + idx
            for idx, line in enumerate(lines)
            if "```" in line[line1_idx + 1 :]
        ]
        if line2_idx:
            line2_idx = line2_idx[0]
            text = "\n".join(lines[:line1_idx]) + "\n".join(lines[line2_idx + 1 :])
        else:
            text = "\n".join(lines[:line1_idx])
        return is_code_conversation(text)

    if "```markdown" in text:
        otext = text
        lines = text.split("\n")
        line1_idx = [idx for idx, line in enumerate(lines) if "```markdown" in line][0]
        line2_idx = [
            line1_idx + 1 + idx
            for idx, line in enumerate(lines)
            if "```" in line[line1_idx + 1 :]
        ]
        if line2_idx:
            line2_idx = line2_idx[0]
            text = "\n".join(lines[:line1_idx]) + "\n".join(lines[line2_idx + 1 :])
        else:
            text = "\n".join(lines[:line1_idx])
        return is_code_conversation(text)

    if "ascii art" in text.lower():
        return False, []

    # 1. Check for code formatting
    if re.search(r"```", text):
        return True, ["backticks"]

    # Tokenize the text
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]

    # 2. Check for programming concepts
    concepts = ["git", "github", "pull request", "dataframe", "nginx", "pip"]
    if any(concept in tokens for concept in concepts):
        matched_concepts = list(set(tokens).intersection(set(concepts)))
        return True, matched_concepts

    # 3. Check for programming language name
    languages = [
        "python",
        "c++",
        "cpp",
        "java",
        "javascript",
        "typescript",
        "html",
        "css",
        "sql",
        "bash",
        "powershell",
        "matlab",
        "golang",
        "linux",
        "ubuntu",
    ]
    if any(language in tokens for language in languages):
        matched_languages = list(set(tokens).intersection(set(languages)))
        return True, matched_languages

    # 4. Programming concept substrings
    strings = [
        "import pandas",
        "import numpy",
        "import torch",
        "jax",
        "tensorflow",
        "pytorch",
        "keras",
        "scikit-learn",
        "sklearn",
        " apt-get ",
    ]
    found_array = [string in text for string in strings]
    if any(found_array):
        matched_strings = [
            string for string, found in zip(strings, found_array) if found
        ]
        return True, matched_strings

    # 5. Programming concept regexes
    regexes = [
        r"from \w+ import \w+",
        r"conda install \w+",
        r"pip install -r \w+",
        r"conda install -c \w+ \w+",
        r"#include <\w+>",
        r"import \w+ as \w+",
        r"#include \"\w+\.h\"",
    ]
    found_array = [re.search(regex, text) for regex in regexes]
    if any(found_array):
        matched_regexes = [regex for regex, found in zip(regexes, found_array) if found]
        return True, matched_regexes

    return False, []


def check_code_conv(conv) -> tuple[bool, list[str]]:
    """Check if the conversation is a code conversation"""
    for _, msg in enumerate(conv):
        content = msg["content"]
        if not isinstance(content, str):
            continue
        is_code_conv_res = is_code_conversation(content)
        if is_code_conv_res[0]:
            return is_code_conv_res
    return False, []


def check_conv_row(conv_row):
    check_a, code_a = check_code_conv(conv_row["conversation_a"])
    check_b, code_b = check_code_conv(conv_row["conversation_b"])

    return check_a or check_b, code_a + code_b


def process_battle_file(battle_file_path: str, n_cpus: int):
    with open(battle_file_path, "r") as f:
        data = json.load(f)

    with mp.Pool(n_cpus) as pool:
        tagged_data = list(tqdm(pool.imap(check_conv_row, data), total=len(data)))

    output_data = [row for row, (is_code, _) in zip(data, tagged_data) if is_code]

    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-battle-file", type=str)
    parser.add_argument("--output-clean-battle-file", type=str, default=None)
    parser.add_argument("--n-cpus", type=int, default=-1)

    args = parser.parse_args()

    if args.output_clean_battle_file is None:
        args.output_clean_battle_file = args.clean_battle_file

    if args.n_cpus == -1:
        args.n_cpus = mp.cpu_count()

    print(
        f"Processing {args.clean_battle_file} and saving to {args.output_clean_battle_file} with {args.n_cpus} cpus"
    )

    output_data = process_battle_file(args.clean_battle_file, args.n_cpus)

    with open(args.output_clean_battle_file, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Total code conversations: {len(output_data)}")
    print("Done!")

    with open(args.output_clean_battle_file, "r") as f:
        data = json.load(f)
