from datasets import load_dataset
import pandas as pd
import ast
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
    "Yoruba": "yo"
}

ENTITIES_TO_IGNORE = ['Organization', 'PersonType', 'DateTime', 'Person', 'Quantity']

def read_credentials(file_path):
    credentials = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Splitting each line by the '=' character
            key, value = line.strip().split('=')
            key = key.strip()
            value = value.strip().strip('"')  # Removing any extra whitespace and quotation marks
            credentials[key] = value
    return credentials

# Usage
credentials = read_credentials('azure_credentials.txt')
key = credentials['key']
endpoint = credentials['endpoint']

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# Authenticate the client using your key and endpoint 
def authenticate_client():
    ta_credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(
            endpoint=endpoint, 
            credential=ta_credential)
    return text_analytics_client

client = authenticate_client()

def redact_text(text, entities):
    new_text = text
    for entity in entities:
        if entity["category"] in ENTITIES_TO_IGNORE:
            print(f"Skipping entity: {entity['category']} = {entity['entity']}")
            continue
        start = entity["offset"]
        end = entity["offset"] + entity["length"]
        new_text = new_text[:start] + "#" * (end - start) + new_text[end:]
    return new_text


# Function to redact PII and return entities in a single text
def redact_pii_and_extract_entities(text, client, language="en"):
    response = client.recognize_pii_entities([text], language=language)
    result = [doc for doc in response if not doc.is_error]

    if result:
        # redacted_text = result[0].redacted_text
        entities = [{'category': entity.category, 'entity': entity.text, 'offset': entity.offset, 'length': entity.length, 'confidence_score': entity.confidence_score} 
                    for entity in result[0].entities]
        redacted_text = redact_text(text, entities) # redact text from entities we want to remove
        return redacted_text, entities
    else:
        return text, []

# Function to redact PII in user responses and extract entities
def redact_user_pii_and_extract_entities(conversation, client, language="en"):
    all_entities = []
    new_conversation = copy.deepcopy(conversation)
    for message in new_conversation:
        if message['role'] == 'user':
            redacted_text, entities = redact_pii_and_extract_entities(message['content'], client, language=language)
            message['content'] = redacted_text
            all_entities.extend(entities)
    return new_conversation, all_entities

def remove_toxic_rows(df):
    toxic_rows = []
    for index, row in df.iterrows():
        if remove_toxic_chat(row['toxic_chat_tag']):
            toxic_rows.append(index)
        elif any([v for k,v in row['openai_moderation']['categories'].items()]):
            toxic_rows.append(index)
    return df.drop(toxic_rows)

def remove_toxic_chat(toxic_chat_tag):
    for k in toxic_chat_tag.keys():
        if toxic_chat_tag[k]['flagged'] :
                return True
    return False

# If the dataset is gated/private, make sure you have run huggingface-cli login
dataset = load_dataset("lmsys/chatbot_arena_conversations")
df = pd.DataFrame(dataset['train'])
old_df_len = len(df)
df = remove_toxic_rows(df)[:10]
print(f"Removed {old_df_len - len(df)} toxic rows")

new_columns = {"redacted_conversation_a": [], "entities_a": [], "redacted_conversation_b": [], "entities_b": [], "unknown_language": []}
for i, row in df.iterrows():
    if language_codes[row['language']] != "n/a":
        redacted_conversation, entities = redact_user_pii_and_extract_entities(row['conversation_a'], client, language_codes[row['language']])
        new_columns["redacted_conversation_a"].append(redacted_conversation)
        new_columns["entities_a"].append(entities)
        redacted_conversation, entities = redact_user_pii_and_extract_entities(row['conversation_b'], client, language_codes[row['language']])
        new_columns["redacted_conversation_b"].append(redacted_conversation)
        new_columns["entities_b"].append(entities)
        new_columns["unknown_language"].append(False) 
    else:
        new_columns["redacted_conversation_a"].append(row['conversation_a'])
        new_columns["entities_a"].append([])
        new_columns["redacted_conversation_b"].append(row['conversation_b'])
        new_columns["entities_b"].append([])
        new_columns["unknown_language"].append(True)
        print(f"Unknown language: {row['language']}")

for k in new_columns:
    df[k] = new_columns[k]

df.to_csv('redacted_conversations-test.csv', index=False)
