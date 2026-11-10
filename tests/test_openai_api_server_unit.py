"""Unit tests for openai_api_server request handling that do not need a running worker."""
import asyncio

from fastchat.protocol.openai_api_protocol import CompletionRequest
from fastchat.serve import openai_api_server as server


def test_completion_with_null_max_tokens(monkeypatch):
    async def check_model(request):
        return None

    async def get_worker_address(model_name):
        return "http://worker"

    async def fetch_remote(url, pload=None, name=None):
        return {"context_length": 2048, "count": 10}[name]

    monkeypatch.setattr(server, "check_model", check_model)
    monkeypatch.setattr(server, "get_worker_address", get_worker_address)
    monkeypatch.setattr(server, "fetch_remote", fetch_remote)

    request = CompletionRequest(model="m", prompt="hi", max_tokens=None, stream=True)
    response = asyncio.run(server.create_completion(request))

    assert request.max_tokens == 2038
    assert response.media_type == "text/event-stream"
