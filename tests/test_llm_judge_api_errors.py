from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import openai
import pytest

from fastchat.conversation import get_conv_template
from fastchat.llm_judge import common
from fastchat.llm_judge import gen_api_answer


@pytest.fixture
def api_server() -> Iterator[tuple[str, list[tuple[int, dict[str, Any]]], list[str]]]:
    responses: list[tuple[int, dict[str, Any]]] = []
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            requests.append(self.path)
            self.rfile.read(int(self.headers["Content-Length"]))
            status, body = responses.pop(0)
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", responses, requests
    server.shutdown()
    thread.join()
    server.server_close()


@pytest.fixture
def api_config(
    monkeypatch: pytest.MonkeyPatch,
    api_server: tuple[str, list[tuple[int, dict[str, Any]]], list[str]],
) -> tuple[list[tuple[int, dict[str, Any]]], list[str]]:
    base_url, responses, requests = api_server
    monkeypatch.setattr(openai, "api_base", base_url)
    monkeypatch.setattr(openai, "api_key", "local-test-key")
    monkeypatch.setattr(openai, "api_type", "open_ai")
    monkeypatch.setattr(openai, "api_version", None)
    monkeypatch.setattr(common, "API_MAX_RETRY", 2)
    monkeypatch.setattr(common, "API_RETRY_SLEEP", 0)
    return responses, requests


def test_context_limit_stops_judgment_without_writing_score(
    api_config: tuple[list[tuple[int, dict[str, Any]]], list[str]], tmp_path: Path
) -> None:
    responses, requests = api_config
    responses.extend(
        [
            (
                400,
                {
                    "error": {
                        "message": "This model's maximum context length is 8192 tokens.",
                        "type": "invalid_request_error",
                        "param": "messages",
                        "code": "context_length_exceeded",
                    }
                },
            )
        ]
        * 2
    )
    match = common.MatchSingle(
        question={"question_id": 1, "turns": ["Question"]},
        model="candidate",
        answer={"choices": [{"turns": ["Answer"]}]},
        judge=common.Judge(
            model_name="gpt-4",
            prompt_template={
                "type": "single",
                "name": "test",
                "system_prompt": "Judge the answer.",
                "prompt_template": "{question} {answer}",
                "output_format": "[[rating]]",
            },
        ),
    )
    output_file = tmp_path / "judgment.jsonl"

    with pytest.raises(
        openai.error.InvalidRequestError, match="maximum context length"
    ):
        common.play_a_match_single(match=match, output_file=str(output_file))

    assert len(requests) == 1
    assert not output_file.exists()


def test_context_limit_stops_answer_generation(
    api_config: tuple[list[tuple[int, dict[str, Any]]], list[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses, requests = api_config
    responses.extend(
        [
            (200, {"choices": [{"message": {"content": "First answer"}}]}),
            (400, {"error": {"message": "maximum context length exceeded"}}),
            (400, {"error": {"message": "maximum context length exceeded"}}),
        ]
    )
    monkeypatch.setattr(
        gen_api_answer, "args", SimpleNamespace(force_temperature=None), raising=False
    )
    output_file = tmp_path / "answer.jsonl"

    with pytest.raises(
        openai.error.InvalidRequestError, match="maximum context length"
    ):
        gen_api_answer.get_answer(
            question={
                "question_id": 1,
                "category": "writing",
                "turns": ["First", "Second"],
            },
            model="gpt-4",
            num_choices=1,
            max_tokens=20,
            answer_file=str(output_file),
        )

    assert len(requests) == 2
    assert not output_file.exists()


@pytest.mark.parametrize("azure", [False, True])
def test_invalid_request_is_not_retried(
    api_config: tuple[list[tuple[int, dict[str, Any]]], list[str]], azure: bool
) -> None:
    responses, requests = api_config
    responses.extend(
        [
            (
                400,
                {
                    "error": {
                        "message": "Invalid request",
                        "type": "invalid_request_error",
                    }
                },
            )
        ]
        * 2
    )
    conv = get_conv_template("chatgpt")
    conv.append_message(conv.roles[0], "Hello")
    conv.append_message(conv.roles[1], None)
    completion = (
        common.chat_completion_openai_azure if azure else common.chat_completion_openai
    )
    kwargs = {"api_base": openai.api_base, "api_key": openai.api_key} if azure else None

    with pytest.raises(openai.error.InvalidRequestError, match="Invalid request"):
        completion(
            model="gpt-4", conv=conv, temperature=0, max_tokens=20, api_dict=kwargs
        )

    assert len(requests) == 1


@pytest.mark.parametrize("azure", [False, True])
def test_rate_limit_is_retried(
    api_config: tuple[list[tuple[int, dict[str, Any]]], list[str]], azure: bool
) -> None:
    responses, requests = api_config
    responses.extend(
        [
            (429, {"error": {"message": "Rate limit", "type": "rate_limit_error"}}),
            (200, {"choices": [{"message": {"content": "[[8]]"}}]}),
        ]
    )
    conv = get_conv_template("chatgpt")
    conv.append_message(conv.roles[0], "Hello")
    conv.append_message(conv.roles[1], None)
    completion = (
        common.chat_completion_openai_azure if azure else common.chat_completion_openai
    )
    kwargs = {"api_base": openai.api_base, "api_key": openai.api_key} if azure else None

    assert (
        completion(
            model="gpt-4", conv=conv, temperature=0, max_tokens=20, api_dict=kwargs
        )
        == "[[8]]"
    )
    assert len(requests) == 2
