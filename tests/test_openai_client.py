import asyncio

from fastchat.client import openai_api_client as client


def test_chat_completion():
    model = "vicuna-7b-v1.1"

    completion = client.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hello! How can I help you today?"},
            {"role": "user", "content": "What is the color of the sky?"},
        ],
    )

    print(completion.choices[0].message.content)


def test_chat_completion_stream():
    model = "vicuna-7b-v1.1"

    res = client.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": "Tell me a story with more than 1000 words."}],
        temperature=0.0,
        max_tokens=128,
        stream=True,
    )

    for chunk in res:
        content = chunk.choices[0].delta.content
        if content is not None:
            print(content, end="", flush=True)
    print()


async def test_chat_completion_stream_async():
    model_name = "vicuna-7b-v1.1"

    res = await client.ChatCompletion.acreate(
        model=model_name,
        messages=[{"role": "user", "content": "Tell me a story with more than 1000 words."}],
        temperature=0.0,
        max_tokens=128,
        stream=True,
    )

    async for chunk in res:
        content = chunk.choices[0].delta.content
        if content is not None:
            print(content, end="", flush=True)
    print()


if __name__ == "__main__":
    test_chat_completion()
    test_chat_completion_stream()
    asyncio.run(test_chat_completion_stream_async())
