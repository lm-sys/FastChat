import os

from fastchat.model import get_conversation_template


def claude():
    import anthropic
    c = anthropic.Client(os.environ["ANTHROPIC_API_KEY"])

    model = "claude-v1"
    conv = get_conversation_template(model)
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    response = c.completion_stream(
        prompt=prompt,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        max_tokens_to_sample=256,
        model=model,
        stream=True,
    )
    for data in response:
        print(data["completion"])


claude()
