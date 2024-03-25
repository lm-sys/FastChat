# A JSON logger that sends data to remote endpoint.
# Architecturally, it hosts a background thread that sends logs to a remote endpoint.
import os
import json
import requests
import threading
import queue
import logging

_global_logger = None


def get_remote_logger():
    global _global_logger
    if _global_logger is None:
        if url := os.environ.get("REMOTE_LOGGER_URL"):
            logging.info(f"Remote logger enabled, sending data to {url}")
            _global_logger = RemoteLogger(url=url)
        else:
            _global_logger = EmptyLogger()
    return _global_logger


class EmptyLogger:
    """Dummy logger that does nothing."""

    def __init__(self):
        pass

    def log(self, _data: dict):
        pass


class RemoteLogger:
    """A JSON logger that sends data to remote endpoint."""

    def __init__(self, url: str):
        self.url = url

        self.logs = queue.Queue()
        self.thread = threading.Thread(target=self._send_logs, daemon=True)
        self.thread.start()

    def log(self, data: dict):
        self.logs.put_nowait(data)

    def _send_logs(self):
        while True:
            data = self.logs.get()

            # process the data by keep only the top level fields, and turn any nested dict into a string
            for key, value in data.items():
                if isinstance(value, (dict, list, tuple)):
                    data[key] = json.dumps(value, ensure_ascii=False)

            try:
                requests.post(self.url, json=data)
            except Exception:
                logging.exception("Failed to send logs to remote endpoint")
