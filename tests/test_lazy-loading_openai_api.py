"""
Use the OpenAI compatible server to test multi_model_worker lazy-loading.

Launch:
python3 launch_lazy-loading_openai_api_test_server.py
"""
import torch

from fastchat.utils import run_cmd
from openai import OpenAI


GIB = 1073741824
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")


def test_list_models():
    model_list = client.models.list()
    names = [x.id for x in model_list.data]
    return names


def test_completion(model):
    prompt = "Once upon a time"
    completion = client.completions.create(
        model=model,
        prompt=prompt,
        logprobs=None,
        max_tokens=64,
        temperature=0,
    )
    print(f"full text: {prompt + completion.choices[0].text}", flush=True)


def test_completion_stream(model):
    prompt = "Once upon a time"
    res = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=64,
        stream=True,
        temperature=0,
    )
    print(prompt, end="")
    for chunk in res:
        content = chunk.choices[0].text
        print(content, end="", flush=True)
    print()


def test_embedding(model):
    embedding = client.embeddings.create(model=model, input="Hello world!")
    print(f"embedding len: {len(embedding.data[0].embedding)}")
    print(f"embedding value[:5]: {embedding.data[0].embedding[:5]}")


def test_chat_completion(model):
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Hello! What is your name?"}],
        temperature=0,
    )
    print(completion.choices[0].message.content)


def test_chat_completion_stream(model):
    messages = [{"role": "user", "content": "Hello! What is your name?"}]
    res = client.chat.completions.create(
        model=model, messages=messages, stream=True, temperature=0
    )
    for chunk in res:
        content = chunk.choices[0].delta.content or ""
        print(content, end="", flush=True)
    print()


def test_openai_curl(model):
    run_cmd("curl http://localhost:8000/v1/models")

    run_cmd(
        """
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": """ f"\"{model}\"" """,
    "messages": [{"role": "user", "content": "Hello! What is your name?"}]
  }'
"""
    )

    run_cmd(
        """
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": """ f"\"{model}\"" """,
    "prompt": "Once upon a time",
    "max_tokens": 41,
    "temperature": 0.5
  }'
"""
    )

    run_cmd(
        """
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": """ f"\"{model}\"" """,
    "input": "Hello world!"
  }'
"""
    )


def model_test_suite(model):
    print(f"===== Test {model} ======")

    test_completion(model)
    test_completion_stream(model)
    test_chat_completion(model)
    test_chat_completion_stream(model)
    try:
        test_embedding(model)
    except client.error.APIError as e:
        print(f"Embedding error: {e}")

    print("===== Test curl with {model} =====")
    test_openai_curl(model)


if __name__ == "__main__":
    models = test_list_models()
    print(f"models: {models}")

    # The launch script specifies "--limit-worker-concurrency 1" so only one
    # model will be loaded into VRAM at a time.

    # LLaMA 1.3B should have been pre-loaded into VRAM, but querying it here
    # ensures LLaMA 1.3B is the loaded model. Check functionality.
    model_test_suite("Sheared-LLaMA-1.3B")
    llama_vram_info = torch.cuda.mem_get_info()
    llama_vram_usage = llama_vram_info[1] - llama_vram_info[0]
    print("LLaMA-1.3B VRAM usage: {:.2f}GiB".format(llama_vram_usage / GIB))

    # Lazy-load fastchat 3B into VRAM. Because --limit-worker-concurrency is
    # set to 1, this will also unload LLaMA 1.3B. Check functionality.
    model_test_suite("fastchat-t5-3b-v1.0")
    fastchat_vram_info = torch.cuda.mem_get_info()
    fastchat_vram_usage = fastchat_vram_info[1] - fastchat_vram_info[0]
    print("fastchat-t5-3b VRAM usage: {:.2f}GiB".format(
        fastchat_vram_usage / GIB))

    # Check that fastchat VRAM usage is significanly higher than LLaMA.
    assert (llama_vram_usage / fastchat_vram_usage) < 0.9

    # Lazy-load LLaMA 1.3B again. Because --limit-worker-concurrency is
    # set to 1, this will also unload fastchat 3B. Check functionality.
    model_test_suite("Sheared-LLaMA-1.3B")
    llama_vram_info = torch.cuda.mem_get_info()
    llama_vram_usage = llama_vram_info[1] - llama_vram_info[0]
    print("LLaMA-1.3B VRAM usage: {:.2f}GiB".format(llama_vram_usage / GIB))

    # Only LLaMA 1.3B should be loaded, so usage should be much lower again.
    assert (llama_vram_usage / fastchat_vram_usage) < 0.9
