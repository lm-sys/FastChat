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
                        {"type": "text", "text": f"Tell me a story about this image."},
                    ],
                },
            ]
        else:
            messages = [{"role": "user", "content": f"Tell me a story about this image."}]

        start = time.time()
        response = await litellm_client.chat.completions.create(
            model=args.model,
            messages=messages,
            stream=True,
        )
        ttft = time.time() - start

        itl_list = []
        content = ""
        start = time.time()
        async for chunk in response:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
                itl_list.append(time.time() - start)
                start = time.time()
        
        return content, ttft, itl_list

    except Exception as e:
        # If there's an exception, log the error message
        print(e)
        with open("error_log.txt", "a") as error_log:
            error_log.write(f"Error during completion: {str(e)}\n")
        return str(e)


async def main(args):
    n = 50 # Total number of tasks
    batch_size = args.req_per_sec  # Requests per second
    start = time.time()

    all_tasks = []
    for i in range(0, n, batch_size):
        batch = range(i, min(i + batch_size, n))
        for _ in batch:
            if args.include_image:
                # Generate a random dimension for the image
                y_dimension = np.random.randint(100, 1025)
                image_url = f"https://placehold.co/1024x{y_dimension}/png"
                task = asyncio.create_task(litellm_completion(args, image_url))
            else:
                task = asyncio.create_task(litellm_completion(args))
            all_tasks.append(task)
        if i + batch_size < n:
            await asyncio.sleep(1)  # Wait 1 second before the next batch

    all_completions = await asyncio.gather(*all_tasks)

    successful_completions = [c for c in all_completions if isinstance(c, tuple) and len(c) == 3]
    ttft_list = np.array([float(c[1]) for c in successful_completions])
    itl_list_flattened = np.array([float(item) for sublist in [c[2] for c in successful_completions] for item in sublist])

    # Write errors to error_log.txt
    with open("error_log.txt", "a") as error_log:
        for completion in all_completions:
            if isinstance(completion, str):
                error_log.write(completion + "\n")

    print(f"Completed requests: {len(successful_completions)}")
    print(f"P99 TTFT: {np.percentile(ttft_list, 99)}")
    print(f"Mean TTFT: {np.mean(ttft_list)}")
    print(f"P99 ITL: {np.percentile(itl_list_flattened, 99)}")
    print(f"Mean ITL: {np.mean(itl_list_flattened)}")


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
