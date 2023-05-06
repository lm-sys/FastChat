import asyncio

from fastchat import client


async def test_async():
    # TODO: Handle failuers
    res = await client.ChatCompletion.acreate(
        model="stable-vicuna-13b",
        messages=[{"role": "user", "content": "How are you?"}],
        stream=True,
    )
    async for chunk in res:
        print(chunk)


def test_sync():
    # TODO: Handle failuers
    res = client.ChatCompletion.create(
        model="stable-vicuna-13b",
        messages=[{"role": "user", "content": "How are you?"}],
        stream=True,
    )
    for chunk in res:
        print(chunk)


# asyncio.run(test_async())
test_sync()
