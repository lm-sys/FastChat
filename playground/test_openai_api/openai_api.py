import os

from fastchat.model import get_conversation_template

def chatgpt():
    import openai
    model = "gpt-3.5-turbo"
    conv = get_conversation_template(model)
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], None)

    messages = conv.to_openai_api_messages()
    print(messages)

    res = openai.ChatCompletion.create(model=model, messages=messages)
    msg = res["choices"][0]["message"]["content"]
    print(msg)

    res = openai.ChatCompletion.create(model=model, messages=messages, stream=True)
    msg = ""
    for chunk in res:
        msg += chunk["choices"][0]["delta"].get("content", "")
    print(msg)


chatgpt()
