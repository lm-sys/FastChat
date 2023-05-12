import openai

openai.api_key = "EMPTY" # Not support yet
openai.api_base = "http://localhost:8000/v1"

model = "vicuna-7b-v1.1"


def test_completion():
    completion = openai.Completion.create(model=model, prompt="Once upon a time", max_tokens=64)
    print(completion.choices[0].text)


def test_chat_completion():
    completion = openai.ChatCompletion.create(
      model=model,
      messages=[{"role": "user", "content": "Hello! What is your name?"}]
    )
    print(completion.choices[0].message.content)


def test_chat_completion_stream():
    messages=[{"role": "user", "content": "Hello! What is your name?"}]
    res = openai.ChatCompletion.create(model=model, messages=messages, stream=True)
    for chunk in res:
        content = chunk["choices"][0]["delta"].get("content", "")
        print(content, end="", flush=True)


if __name__ == "__main__":
    test_completion()
    test_chat_completion()
    test_chat_completion_stream()
