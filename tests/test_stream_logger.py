import logging

from fastchat.utils import infer_stream_log_level


def test_infer_stream_log_level_maps_uvicorn_prefixes():
    assert (
        infer_stream_log_level("INFO:     Started server", logging.ERROR)
        == logging.INFO
    )
    assert (
        infer_stream_log_level("WARNING: deprecated", logging.ERROR) == logging.WARNING
    )
    assert infer_stream_log_level("ERROR: boom", logging.ERROR) == logging.ERROR
    assert infer_stream_log_level("plain stderr line", logging.ERROR) == logging.ERROR
