import asyncio

from fastchat import client


async def test():
    res = await client.ChatCompletion.acreate(
        model="vicuna-13b-v1.1",
        messages=[{
            "role": "user",
            "content": "Context 1: Title: Teammates > Removing teammates Body: To remove a teammate from your Yotpo account: 1) Log in to your Yotpo admin. 2) Click the Account icon in the upper right-hand corner of the Yotpo Admin. 3) Click Account Settings. 4) In the Teammates section, hover over the teammate you'd like to remove. 5) Click Remove User. 6) In the confirmation popup, click Remove User to confirm their removal. Please note: Teammates who were removed from your account will be automatically notified by email. Instructions: Answer the following question based on the contexts above. First, generate whether the question is answerable or not by outputting \"answerable\" or \"not answerable\". Then specify a complete answer in case the question is answerable. Question: how do remove a team mate? Answer:"
        }],
        stream=True
    )
    async for chunk in res:
        print(chunk)
    # async for chunk in client.ChatCompletion.acreate(
    #     model="vicuna-13b-v1.1",
    #     messages=[{
    #         "role": "user",
    #         "content": "Context 1: Title: Teammates > Removing teammates Body: To remove a teammate from your Yotpo account: 1) Log in to your Yotpo admin. 2) Click the Account icon in the upper right-hand corner of the Yotpo Admin. 3) Click Account Settings. 4) In the Teammates section, hover over the teammate you'd like to remove. 5) Click Remove User. 6) In the confirmation popup, click Remove User to confirm their removal. Please note: Teammates who were removed from your account will be automatically notified by email. Instructions: Answer the following question based on the contexts above. First, generate whether the question is answerable or not by outputting \"answerable\" or \"not answerable\". Then specify a complete answer in case the question is answerable. Question: how do remove a team mate? Answer:"
    #     }],
    #     stream=True,
    # ):
    #     content = chunk["choices"][0].get("delta", {}).get("content")
    #     if content is not None:
    #         print(content, end='')

asyncio.run(test())
