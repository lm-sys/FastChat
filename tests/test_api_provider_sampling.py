import importlib
import json
import logging
from unittest.mock import patch

import httpx
import openai
import pytest


@pytest.fixture
def provider():
    # Import the actual module without redirecting pytest's stdout to a file.
    with patch("fastchat.utils.build_logger", return_value=logging.getLogger(__name__)):
        return importlib.import_module("fastchat.serve.api_provider")


@pytest.fixture
def requests_sent(monkeypatch):
    bodies = []
    clients = []
    real_client = openai.OpenAI

    def respond(request):
        body = json.loads(request.content)
        bodies.append(body)
        choice = {"index": 0, "finish_reason": "stop"}
        response = {"id": "fixture", "created": 0, "model": "fixture-model"}
        if body.get("stream"):
            response.update(
                object="chat.completion.chunk",
                choices=[dict(choice, delta={"content": "ok"})],
            )
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text="data: " + json.dumps(response) + "\n\ndata: [DONE]\n\n",
            )
        response.update(
            object="chat.completion",
            choices=[dict(choice, message={"role": "assistant", "content": "ok"})],
        )
        return httpx.Response(200, json=response)

    def make_client(**kwargs):
        client = real_client(
            **kwargs,
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        )
        clients.append(client)
        return client

    monkeypatch.setattr(openai, "OpenAI", make_client)
    try:
        yield bodies
    finally:
        for client in clients:
            client.close()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("top_p", [0.2, 0.9])
def test_openai_provider_forwards_top_p(provider, requests_sent, stream, top_p):
    result = list(
        provider.openai_api_stream_iter(
            "fixture-model",
            [{"role": "user", "content": "hi"}],
            0.7,
            top_p,
            16,
            api_base="https://mock.invalid/v1",
            api_key="offline-fixture-key",
            stream=stream,
        )
    )
    assert result[-1] == {"text": "ok", "error_code": 0}
    assert requests_sent[0]["top_p"] == top_p
    assert requests_sent[0]["temperature"] == 0.7
    assert requests_sent[0]["max_tokens"] == 16


def test_o1_provider_keeps_sampling_parameters_omitted(provider, requests_sent):
    result = list(
        provider.openai_api_stream_iter(
            "o1-fixture",
            [{"role": "user", "content": "hi"}],
            0.7,
            0.2,
            16,
            api_base="https://mock.invalid/v1",
            api_key="offline-fixture-key",
            is_o1=True,
        )
    )
    assert result[-1]["text"] == "ok"
    assert "top_p" not in requests_sent[0]
    assert "max_tokens" not in requests_sent[0]
    assert requests_sent[0]["temperature"] == 1.0
