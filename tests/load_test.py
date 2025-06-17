import argparse
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid
import traceback
import numpy as np
from transformers import AutoTokenizer
from litellm import completion


def litellm_completion(args, tokenizer, image_url=None):
    try:
        if image_url:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": "Tell me a story about this image."},
                    ],
                },
            ]
        else:
            messages = [
                {"role": "user", "content": "Tell me a story about this image."}
            ]

        start = time.time()

        additional_api_kwargs = {}
        if args.api_key:
            additional_api_kwargs["api_key"] = args.api_key
        if args.api_base:
            additional_api_kwargs["api_base"] = args.api_base

        response = completion(
            model=args.model,
            messages=messages,
            stream=True,
            **additional_api_kwargs,
        )
        ttft = None

        itl_list = []
        content = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                end_time = time.time()
                if ttft is None:
                    ttft = end_time - start
                content += chunk.choices[0].delta.content
                num_tokens = len(tokenizer.encode(content))
                itl_list.append((end_time - start) / num_tokens)
                start = end_time

        return content, ttft, itl_list

    except Exception as e:
        print(e)
        with open("error_log.txt", "a") as error_log:
            error_log.write(f"Error during completion: {str(e)}\n")
        return str(e)


def main(args):
    n = args.num_total_responses
    batch_size = args.req_per_sec  # Requests per second
    start = time.time()

    all_results = []
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        for i in range(0, n, batch_size):
            batch_futures = []
            batch = range(i, min(i + batch_size, n))

            for _ in batch:
                if args.include_image:
                    if args.randomize_image_dimensions:
                        y_dimension = np.random.randint(100, 1025)
                    else:
                        y_dimension = 512
                    image_url = f"https://placehold.co/1024x{y_dimension}/png"
                    future = executor.submit(
                        litellm_completion, args, tokenizer, image_url
                    )
                else:
                    future = executor.submit(litellm_completion, args, tokenizer)
                batch_futures.append(future)

            # Wait for batch to complete
            for future in batch_futures:
                all_results.append(future.result())

            if i + batch_size < n:
                time.sleep(1)  # Wait 1 second before next batch

    successful_completions = [
        c for c in all_results if isinstance(c, tuple) and len(c) == 3
    ]
    ttft_list = np.array([float(c[1]) for c in successful_completions])
    itl_list_flattened = np.array(
        [
            float(item)
            for sublist in [c[2] for c in successful_completions]
            for item in sublist
        ]
    )

    # Write errors to error_log.txt
    with open("load_test_errors.log", "a") as error_log:
        for completion in all_results:
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
    parser.add_argument("--api-base", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--num-total-responses", type=int, default=50)
    parser.add_argument("--req-per-sec", type=int, default=5)
    parser.add_argument("--include-image", action="store_true")
    parser.add_argument("--randomize-image-dimensions", action="store_true")
    args = parser.parse_args()

    # Blank out contents of error_log.txt
    open("load_test_errors.log", "w").close()

    main(args)
