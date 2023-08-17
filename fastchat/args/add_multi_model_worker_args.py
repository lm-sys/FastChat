def add_multi_model_worker_args(parser):
    # Note: Ensure we resolve arg conflicts.  We let `add_model_args` add MOST
    # of the model args but we'll override one to have an append action that
    # supports multiple values.
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    # Override the model path to be repeated and align it with model names.
    parser.add_argument(
        "--model-path",
        type=str,
        default=[],
        action="append",
        help="One or more paths to model weights to load. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        action="append",
        help="One or more model names.  Values must be aligned with `--model-path` values.",
    )
    parser.add_argument("--limit-worker-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")