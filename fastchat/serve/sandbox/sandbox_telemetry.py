'''
Module for logging the sandbox interactions and state.

TODO: Support Cloud Storage.
'''
from datetime import datetime
import json
import os
from typing import Any

from fastchat.constants import LOGDIR
from fastchat.serve.sandbox.code_runner import ChatbotSandboxState


def get_sandbox_log_filename(sandbox_id: str) -> str:
    t = datetime.now()
    name = os.path.join(LOGDIR, f"sandbox-records-{sandbox_id}.json")
    return name

def upsert_sandbox_log(sandbox_id: str, data: dict):
    filename = get_sandbox_log_filename(sandbox_id)
    with open(filename, "w") as fout:
        json.dump(
            data,
            fout,
            indent=2,
            default=str,
            ensure_ascii=False
        )

def create_sandbox_log(sandbox_state: ChatbotSandboxState, user_interaction_records: list[Any]) -> dict:
    return {
        "sandbox_state": sandbox_state,
        "user_interaction_records": user_interaction_records,
    }

def log_sandbox_telemetry_gradio_fn(
    sandbox_state: ChatbotSandboxState,
    sandbox_ui_value: tuple[str, bool, list[Any]]
) -> None:
    sandbox_id = sandbox_state['sandbox_id']
    user_interaction_records = sandbox_ui_value[2]
    if sandbox_id and user_interaction_records and len(user_interaction_records) > 0:
        data = create_sandbox_log(sandbox_state, user_interaction_records)
        upsert_sandbox_log(sandbox_id, data)