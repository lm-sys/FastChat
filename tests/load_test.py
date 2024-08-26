import argparse
import time, asyncio
from openai import AsyncOpenAI, AsyncAzureOpenAI
import uuid
import traceback
import numpy as np

# base_url - litellm proxy endpoint
# api_key - litellm proxy api-key, is created proxy with auth
litellm_client = None


async def litellm_completion(args, image_url=None):
    # Your existing code for litellm_completion goes here
    try:
        if image_url:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": f"This is a test: {uuid.uuid4()}"},
                    ],
                },
            ]
        else:
            messages = [{"role": "user", "content": f"This is a test: {uuid.uuid4()}"}]

        response = await litellm_client.chat.completions.create(
            model=args.model,
            messages=messages,
        )
        print(response)
        return response

    except Exception as e:
        # If there's an exception, log the error message
        with open("error_log.txt", "a") as error_log:
            error_log.write(f"Error during completion: {str(e)}\n")
        return str(e)


async def main(args):
    n = 100  # Total number of tasks
    batch_size = args.req_per_sec  # Requests per second
    start = time.time()

    async def run_batch(batch):
        tasks = []
        for _ in batch:
            if args.include_image:
                # Generate a random dimension for the image
                y_dimension = np.random.randint(100, 1025)
                image_url = f"https://placehold.co/1024x{y_dimension}/png"
                task = litellm_completion(args, image_url)
            else:
                task = litellm_completion(args)
            tasks.append(task)
        return await asyncio.gather(*tasks)

    all_completions = []
    for i in range(0, n, batch_size):
        batch = range(i, min(i + batch_size, n))
        print("Starting to run on batch number: {}".format(i))
        completions = await run_batch(batch)
        all_completions.extend(completions)
        if i + batch_size < n:
            await asyncio.sleep(1)  # Wait 1 second before the next batch

    successful_completions = [c for c in all_completions if c is not None]

    # Write errors to error_log.txt
    with open("error_log.txt", "a") as error_log:
        for completion in all_completions:
            if isinstance(completion, str):
                error_log.write(completion + "\n")

    print(n, time.time() - start, len(successful_completions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="azure-gpt-3.5")
    parser.add_argument("--server-address", type=str, default="http://0.0.0.0:9094")
    parser.add_argument("--req-per-sec", type=int, default=5)
    parser.add_argument("--include-image", action="store_true")
    args = parser.parse_args()

    litellm_client = AsyncOpenAI(base_url=args.server_address, api_key="sk-1234")
    # Blank out contents of error_log.txt
    open("error_log.txt", "w").close()

    asyncio.run(main(args))
