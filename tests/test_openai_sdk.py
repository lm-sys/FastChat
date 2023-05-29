import openai

openai.api_key = "EMPTY"  # Not support yet
openai.api_base = "http://localhost:8000/v1"

model = "vicuna-7b-v1.1"


def test_list_models():
    model_list = openai.Model.list()
    print(model_list["data"][0]["id"])


def test_completion():
    prompt = "Once upon a time"
    completion = openai.Completion.create(model=model, prompt=prompt, max_tokens=64)
    print(prompt + completion.choices[0].text)


def test_embedding():
    embedding = openai.Embedding.create(model=model, input="Hello world!")
    print(len(embedding["data"][0]["embedding"]))


def test_chat_completion():
    completion = openai.ChatCompletion.create(
        model=model, messages=[{"role": "user", "content": "Hello! What is your name?"}]
    )
    print(completion.choices[0].message.content)


def test_chat_completion_stream():
    messages = [{"role": "user", "content": "Hello! What is your name?"}]
    res = openai.ChatCompletion.create(model=model, messages=messages, stream=True)
    for chunk in res:
        content = chunk["choices"][0]["delta"].get("content", "")
        print(content, end="", flush=True)
    print()


if __name__ == "__main__":
    test_list_models()
    test_completion()
    test_embedding()
    test_chat_completion()
    test_chat_completion_stream()
