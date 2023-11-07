from fastchat.conversation import get_conv_template


def get_sample_conversation() -> list[str]:
    return [
        "What is your favourite condiment?",
        "Well, I'm quite partial to a good squeeze of fresh lemon juice. "
        "It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!",
        "Do you have mayonnaise recipes?",
        "Here is a recipe for mayonnaise.",
    ]


def test_chat_template_mistral():
    conversation = get_sample_conversation()

    conv_template = get_conv_template("mistral")
    conv_template.append_message(conv_template.roles[0], conversation[0])
    conv_template.append_message(conv_template.roles[1], conversation[1])
    conv_template.append_message(conv_template.roles[0], conversation[2])
    conv_template.append_message(conv_template.roles[1], conversation[3])
    prompt = conv_template.get_prompt()

    expected_prompt = (
        f"<s> [INST] {conversation[0]} [/INST]{conversation[1]}</s>  "
        f"[INST] {conversation[2]} [/INST]"
        f"{conversation[3]}</s> "
    )

    assert prompt == expected_prompt


if __name__ == "__main__":
    test_chat_template_mistral()
