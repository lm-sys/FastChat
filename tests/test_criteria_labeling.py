"""Tests for fastchat.serve.monitor.criteria_labeling.

Run with `pytest tests/test_criteria_labeling.py`.
"""

from unittest import mock

import pytest

pytest.importorskip("openai")
pytest.importorskip("pandas")
pytest.importorskip("tqdm")

from fastchat.serve.monitor import criteria_labeling


def test_chat_completion_openai_handles_key_error():
    client = mock.MagicMock()
    client.chat.completions.create.side_effect = KeyError("choices")

    with mock.patch("openai.OpenAI", return_value=client):
        output = criteria_labeling.chat_completion_openai(
            "model", [{"role": "user", "content": "hi"}], 0.0, 16
        )

    assert output == criteria_labeling.API_ERROR_OUTPUT
    # A KeyError is not retried: the call is made once and the loop breaks.
    assert client.chat.completions.create.call_count == 1
