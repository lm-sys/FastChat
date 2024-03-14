"""
Filter conversations for release.

Dependency:
pip install opencc-python-reimplementedpip install opencc-python-reimplemented

Usage:
python3 filter_bad_conv_lmsys_chat_1m.py --in clean_battle_conv_20230630_tagged_v1_pii.json
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from enum import Enum, auto
import json
import os
import random

from tqdm import tqdm
import opencc

BLOCKED_WORDS_FILENAME = "blocked_words.json"
blocked_words = []
frequency = defaultdict(lambda: 0)

cc_converter = opencc.OpenCC("t2s")


class TypeCode(Enum):
    CORRECT = auto()
    ANONYMIZED = auto()
    REDACTED = auto()
    BAD_FORMAT = auto()
    BLOCKED_WORD = auto()
    BLOCKED_MODEL = auto()
    TOO_SHORT = auto()
    TOO_FREQUENT = auto()


def detect_type(conv):
    for key in ["conversation_a", "conversation_b", "conversation"]:
        if key not in conv:
            continue

        messages = [row["content"] for row in conv[key]]
        for msg in messages:
            if not isinstance(msg, str):
                return TypeCode.BAD_FORMAT

        if len(messages) == 0:
            return TypeCode.BAD_FORMAT

        user_prompts = [
            row["content"].lower().strip() for row in conv[key] if row["role"] == "user"
        ]

        for msg in messages:
            msg = cc_converter.convert(msg.lower())
            if "<anonymized>" in msg:
                return TypeCode.ANONYMIZED
            if "<redacted>" in msg:
                return TypeCode.REDACTED

            for w in blocked_words:
                if w in msg:
                    return TypeCode.BLOCKED_WORD

    return TypeCode.CORRECT


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--sample", type=int)
    args = parser.parse_args()

    # Read conversations
    convs = json.load(open(args.in_file))
    print(f"#conv: {len(convs)}")

    # Read blocked words
    if os.path.exists(BLOCKED_WORDS_FILENAME):
        blocked_words = json.load(open(BLOCKED_WORDS_FILENAME))
        blocked_words = [cc_converter.convert(w) for w in blocked_words]

    # Start filter
    ct_bad_format = 0
    ct_anonymized = 0
    ct_redacted = 0
    ct_error = 0
    ct_lang_filter = 0
    ct_flagged = 0
    ct_blocked_word = 0
    ct_blocked_model = 0
    ct_too_short = 0
    ct_too_frequent = 0

    type_codes = []
    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(detect_type, convs), total=len(convs)):
            type_codes.append(result)

    new_convs = []
    for conv, type_code in zip(convs, type_codes):
        if type_code == TypeCode.BAD_FORMAT:
            ct_bad_format += 1
            continue

        if type_code == TypeCode.ANONYMIZED:
            ct_anonymized += 1
            continue
        elif type_code == TypeCode.REDACTED:
            ct_redacted += 1
            continue
        elif type_code == TypeCode.BLOCKED_WORD:
            ct_blocked_word += 1
            continue
        elif type_code == TypeCode.BLOCKED_MODEL:
            ct_blocked_model += 1
            continue
        elif type_code == TypeCode.TOO_SHORT:
            ct_too_short += 1
            continue
        elif type_code == TypeCode.TOO_FREQUENT:
            ct_too_frequent += 1
            continue

        if "openai_moderation" in conv and conv["openai_moderation"]["flagged"]:
            ct_flagged += 1
            continue

        if type_code in [TypeCode.CORRECT]:
            new_convs.append(conv)

    if args.sample:
        random.seed(42)
        random.shuffle(new_convs)
        new_convs = new_convs[: args.sample]

    print(f"ct_anonymized: {ct_anonymized}, ct_redacted: {ct_redacted}")
    print(f"ct_bad_format: {ct_bad_format}, ct_flagged: {ct_flagged}")
    print(f"ct_blocked_word: {ct_blocked_word}, ct_blocked_model: {ct_blocked_model}")
    print(f"ct_too_short: {ct_too_short}, ct_too_frequent: {ct_too_frequent}")
    print(f"new_conv: {len(new_convs)}")

    out_file = args.in_file.replace(".json", ".s1.json")
    print(f"Output to {out_file}")
    with open(out_file, "w") as fout:
        json.dump(new_convs, fout, indent=2, ensure_ascii=False)
