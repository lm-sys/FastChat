from datasets import load_dataset
import pandas as pd
import ast
import csv
import copy

language_codes = {
    "English": "en",
    "unknown": "n/a",
    "French": "fr",
    "Latin": "la",
    "Arabic": "ar",
    "Italian": "it",
    "Latvian": "lv",
    "Portuguese": "pt",
    "Japanese": "ja",
    "German": "de",
    "Spanish": "es",
    "Chinese": "zh",
    "Russian": "ru",
    "Slovak": "sk",
    "Turkish": "tr",
    "Akan": "n/a",
    "Danish": "da",
    "Uzbek": "uz",
    "Esperanto": "eo",
    "Scots": "sco",
    "Indonesian": "id",
    "Hebrew": "he",
    "Dutch": "nl",
    "Korean": "ko",
    "Corsican": "co",
    "Wolof": "wo",
    "Waray": "n/a",
    "Luxembourgish": "lb",
    "Bulgarian": "bg",
    "Serbian": "sr",
    "Czech": "cs",
    "Catalan": "ca",
    "Manx": "gv",
    "Swedish": "sv",
    "Malagasy": "mg",
    "Polish": "pl",
    "Norwegian": "no",
    "Interlingua": "ia",
    "Oromo": "om",
    "Tswana": "tn",
    "Finnish": "fi",
    "Maori": "mi",
    "Tsonga": "ts",
    "Romanian": "ro",
    "Bislama": "bi",
    "Welsh": "cy",
    "Xhosa": "xh",
    "Galician": "gl",
    "Malay": "ms",
    "Persian": "fa",
    "Vietnamese": "vi",
    "zzp": "n/a",
    "Hawaiian": "haw",
    "Aymara": "ay",
    "Norwegian Nynorsk": "nn",
    "Afrikaans": "af",
    "Icelandic": "is",
    "Ukrainian": "uk",
    "Occitan": "oc",
    "Hungarian": "hu",
    "Thai": "th",
    "Lithuanian": "lt",
    "Croatian": "hr",
    "Quechua": "qu",
    "Haitian Creole": "ht",
    "Western Frisian": "fy",
    "Interlingue": "ie",
    "Somali": "so",
    "Slovenian": "sl",
    "Afar": "aa",
    "Irish": "ga",
    "Kalaallisut": "kl",
    "Volapük": "vo",
    "Fijian": "fj",
    "Sanskrit": "sa",
    "Estonian": "et",
    "Basque": "eu",
    "Hmong": "n/a",
    "Macedonian": "mk",
    "Kinyarwanda": "rw",
    "Bangla": "bn",
    "Southern Sotho": "st",
    "Tatar": "tt",
    "Sundanese": "su",
    "Greek": "el",
    "Guarani": "gn",
    "Klingon": "tlh",
    "Shona": "sn",
    "Hindi": "hi",
    "Maltese": "mt",
    "Ganda": "lg",
    "Swahili": "sw",
    "Cebuano": "ceb",
    "Tigrinya": "ti",
    "Faroese": "fo",
    "Yoruba": "yo",
    "Māori": "n/a",
    "unknown": "en", # this is set to en because we don't know the language
}

ENTITIES_TO_IGNORE = ["Organization", "PersonType", "DateTime", "Person", "Quantity"]


def read_credentials(file_path):
    credentials = {}
    with open(file_path, "r") as file:
        for line in file:
            # Splitting each line by the '=' character
            key, value = line.strip().split("=")
            key = key.strip()
            value = value.strip().strip(
                '"'
            )  # Removing any extra whitespace and quotation marks
            credentials[key] = value
    return credentials


# Usage
credentials = read_credentials("azure_credentials.txt")
key = credentials["key"]
endpoint = credentials["endpoint"]

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential


# Authenticate the client using your key and endpoint
def authenticate_client():
    ta_credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(
        endpoint=endpoint, credential=ta_credential
    )
    return text_analytics_client


client = authenticate_client()


def redact_text(text, entities):
    redacted = False
    new_text = text
    for entity in entities:
        if entity["category"] in ENTITIES_TO_IGNORE:
            # print(f"Skipping entity: {entity['category']} = {entity['entity']}")
            continue
        print(f"Redacting entity: {entity['category']} = {entity['entity']}")
        start = entity["offset"]
        end = entity["offset"] + entity["length"]
        new_text = new_text[:start] + "#" * (end - start) + new_text[end:]
        redacted = True
    return new_text, redacted


# Function to redact PII and return entities in a single text
def redact_pii_and_extract_entities(text, client, language="en"):
    response = client.recognize_pii_entities([text], language=language)
    result = [doc for doc in response if not doc.is_error]

    if result:
        # redacted_text = result[0].redacted_text
        entities = [
            {
                "category": entity.category,
                "entity": entity.text,
                "offset": entity.offset,
                "length": entity.length,
                "confidence_score": entity.confidence_score,
            }
            for entity in result[0].entities
        ]
        redacted_text, redacted = redact_text(
            text, entities
        )  # redact text from entities we want to remove
        return redacted_text, entities, redacted
    else:
        return text, [], False


# Function to redact PII in user responses and extract entities
def redact_user_pii_and_extract_entities(conversation, client, language="en"):
    all_entities = []
    new_conversation = copy.deepcopy(conversation)
    for message in new_conversation:
        if message["role"] == "user":
            redacted_text, entities, redacted = redact_pii_and_extract_entities(
                message["content"], client, language=language
            )
            message["content"] = redacted_text
            all_entities.extend(entities)
    return new_conversation, all_entities, redacted


def remove_toxic_rows(df):
    toxic_rows = []
    for index, row in df.iterrows():
        print(row.keys())
        if remove_toxic_chat(row["toxic_chat_tag"]):
            toxic_rows.append(index)
        elif any([v for k, v in row["openai_moderation"]["categories"].items()]):
            toxic_rows.append(index)
    return df.drop(toxic_rows)


def remove_toxic_chat(toxic_chat_tag):
    for k in toxic_chat_tag.keys():
        if toxic_chat_tag[k]["flagged"]:
            return True
    return False

def replace_user_entries(original, redacted):
    for i, item in enumerate(original):
        if item['role'] == 'user':
            original[i] = redacted[i]
    return original

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_json", type=str, help="Input file path")
parser.add_argument("--csv_output", type=str, help="Output file path")
parser.add_argument("--test", action="store_true", help="Test mode")
args = parser.parse_args()

# If the dataset is gated/private, make sure you have run huggingface-cli login
df = pd.read_json(args.input_json)
old_df_len = len(df)
if args.test:
    df = df.head(50)
output_file = args.input_json.replace(".json", "_redacted.csv") if not args.csv_output else args.csv_output
if args.test:
    output_file = output_file.replace(".csv", "_test.csv")
    
# uncomment if you want to remove toxic rows
# df = remove_toxic_rows(df)
# print(f"Removed {old_df_len - len(df)} toxic rows")

# Define your columns
columns = [
    "redacted_conversation_a",
    "entities",
    "redacted_entities",
    "redacted_conversation_b",
    "pii_detection_succeeded",
    "redacted",
]
columns = list(df.columns) + columns 

# Open a CSV file to write
with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()

    for i, row in df.iterrows():
        csv_row = {}
        for k, v in row.items():
            csv_row[k] = v
        if (
            row["language"] in language_codes.keys()
            and language_codes[row["language"]] != "n/a"
        ):
            redacted_conversation, entities, redacted = redact_user_pii_and_extract_entities(
                row["conversation_a"], client, language_codes[row["language"]]
            )
            csv_row["redacted_conversation_a"] = redacted_conversation
            csv_row["entities"] = entities
            csv_row["redacted_entities"] = [e for e in entities if e["category"] not in ENTITIES_TO_IGNORE]
            csv_row["redacted_conversation_b"] = replace_user_entries(copy.deepcopy(row["conversation_b"]), redacted_conversation)
            csv_row["pii_detection_succeeded"] = True
            csv_row["redacted"] = redacted
        else:
            csv_row["redacted_conversation_a"] = row["conversation_a"]
            csv_row["entities"] = []
            csv_row["redacted_entities"] = []
            csv_row["redacted_conversation_b"] = row["conversation_b"]
            csv_row["pii_detection_succeeded"] = False
            csv_row["redacted"] = False
            print(f"Unknown language: {row['language']}")

        # Write the processed row to the CSV
        writer.writerow(csv_row)
