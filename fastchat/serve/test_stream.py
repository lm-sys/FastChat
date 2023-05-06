import argparse
import asyncio

from fastchat import client


async def async_main():
    # TODO: Handle failuers
    model_name = args.model_name

    res = await client.ChatCompletion.acreate(
        model=model_name,
        messages=[{"role": "user", "content": args.message}],
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        stream=True,
    )
    async for chunk in res:
        content = chunk.choices[0].delta.content
        if content is not None:
            print(content, end="")


def sync_main():
    # TODO: Handle failuers
    model_name = args.model_name

    res = client.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": args.message}],
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        stream=True,
    )
    for chunk in res:
        content = chunk.choices[0].delta.content
        if content is not None:
            print(content, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--worker-address", type=str)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--message", type=str, default="Tell me a story with more than 1000 words."
    )

    parser.add_argument("--run-async", action="store_true")
    parser.add_argument("--no-run-async", dest="run_async", action="store_false")
    parser.set_defaults(run_async=True)

    args = parser.parse_args()

    if args.run_async:
        asyncio.run(async_main())
    else:
        sync_main()
