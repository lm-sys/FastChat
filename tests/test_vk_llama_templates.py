import pytest

from transformers import AutoTokenizer
from fastchat.model import get_conversation_template

MODEL_PATH = # Write the model path here
FIRST_TURN = "Привет! Как я могу стать исследователем в области машинного обучения?"
FIRST_ANSWER = "Вам стоит прочитать много книг по этой теме!"
SECOND_TURN = "А что еще можешь посоветовать?"
DEFAULT_SYSTEM_ROLE = "Ты полезный AI-ассистент."
CUSTOM_SYSTEM_ROLE = "You are a helpful assistant. Answer only in correct and honest way."


def setup_tokenizer_and_conv(model_path: str):
    conv = get_conversation_template(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return tokenizer, conv


def test_single_instruction():
    tokenizer, conv = setup_tokenizer_and_conv(MODEL_PATH)

    conv.append_message(conv.roles[0], FIRST_TURN)
    conv.append_message(conv.roles[1], None)
    fast_chat_prompt = conv.get_prompt()

    hf_input_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": DEFAULT_SYSTEM_ROLE},
            {"role": "user", "content": FIRST_TURN}
        ],
    )

    fast_chat_input_ids = tokenizer([fast_chat_prompt]).input_ids[0]

    assert fast_chat_input_ids == hf_input_ids


def test_dialogue():
    tokenizer, conv = setup_tokenizer_and_conv(MODEL_PATH)

    conv.append_message(conv.roles[0], FIRST_TURN)
    conv.append_message(conv.roles[1], FIRST_ANSWER)
    conv.append_message(conv.roles[0], SECOND_TURN)
    conv.append_message(conv.roles[1], None)
    fast_chat_prompt = conv.get_prompt()

    hf_input_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": DEFAULT_SYSTEM_ROLE},
            {"role": "user", "content": FIRST_TURN},
            {"role": "assistant", "content": FIRST_ANSWER},
            {"role": "user", "content": SECOND_TURN}
        ],
    )

    fast_chat_input_ids = tokenizer([fast_chat_prompt]).input_ids[0]

    assert fast_chat_input_ids == hf_input_ids


def test_system_role():
    tokenizer, conv = setup_tokenizer_and_conv(MODEL_PATH)

    conv.set_system_message(CUSTOM_SYSTEM_ROLE)
    conv.append_message(conv.roles[0], FIRST_TURN)
    conv.append_message(conv.roles[1], FIRST_ANSWER)
    conv.append_message(conv.roles[0], SECOND_TURN)
    conv.append_message(conv.roles[1], None)
    fast_chat_prompt = conv.get_prompt()

    hf_input_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": CUSTOM_SYSTEM_ROLE},
            {"role": "user", "content": FIRST_TURN},
            {"role": "assistant", "content": FIRST_ANSWER},
            {"role": "user", "content": SECOND_TURN}
        ],
    )

    fast_chat_input_ids = tokenizer([fast_chat_prompt]).input_ids[0]

    assert fast_chat_input_ids == hf_input_ids
