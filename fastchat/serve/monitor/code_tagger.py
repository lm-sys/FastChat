import re
import json
import argparse
import multiprocessing as mp

import nltk
from tqdm import tqdm
from nltk.tokenize import word_tokenize



def is_code_conversation(text : str) -> tuple[bool, list[str]]:
    """Check if the text is a code conversation"""

    # 1. Check for code formatting
    if re.search(r'```', text) or re.search(r'^[ \t]+\S', text, re.MULTILINE):
        return True, ['backticks']

    # Tokenize the text
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    

    # 2. Check for programming concepts
    concepts = ["debug", "compile", "git", "github", "api", "package", "trace", "exception", "pull req", "branch", "import", "pandas", "array", "dataframe", "pandas"]
    if any(concept in tokens for concept in concepts):
        matched_concepts = list(set(tokens).intersection(set(concepts)))
        return True, matched_concepts

    # 3. Check for programming language name
    languages = ["python", "c++", 'c', "cpp", "java", "javascript", "typescript", "html", "css","sql", "bash", "shell", "powershell", "matlab", 'r']
    if any(language in tokens for language in languages):
        matched_languages = list(set(tokens).intersection(set(languages)))
        return True, matched_languages

    return False, []


def check_code_conv(conv)  -> tuple[bool, list[str]]:
    """ Check if the conversation is a code conversation"""
    for _, msg in enumerate(conv):
        content = msg['content']
        if not isinstance(content, str):
            continue
        is_code_conv_res = is_code_conversation(content)
        if is_code_conv_res[0]:
            return is_code_conv_res
    return False, []

def check_conv_row(conv_row):
    check_a, code_a = check_code_conv(conv_row['conversation_a']) 
    check_b, code_b = check_code_conv(conv_row['conversation_b'])
    
    return check_a or check_b, code_a + code_b

def process_battle_file(battle_file_path : str, n_cpus: int):
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

    print(f"Processing {args.clean_battle_file} and saving to {args.output_clean_battle_file} with {args.n_cpus} cpus")

    output_data =    process_battle_file(args.clean_battle_file, args.n_cpus)   

    with open(args.output_clean_battle_file, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Total code conversations: {len(output_data)}")
    print("Done!")
    
    with open(args.output_clean_battle_file, "r") as f:
        data = json.load(f)
    
    
