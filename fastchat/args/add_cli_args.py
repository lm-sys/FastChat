def add_cli_args(parser):
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--conv-system-msg", type=str, default=None, help="Conversation system message."
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument(
        "--style",
        type=str,
        default="simple",
        choices=["simple", "rich", "programmatic"],
        help="Display style.",
    )
    parser.add_argument(
        "--multiline",
        action="store_true",
        help="Enable multiline input. Use ESC+Enter for newline.",
    )
    parser.add_argument(
        "--mouse",
        action="store_true",
        help="[Rich Style]: Enable mouse support for cursor positioning.",
    )
    parser.add_argument(
        "--judge-sent-end",
        action="store_true",
        help="Whether enable the correction logic that interrupts the output of sentences due to EOS.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print useful debug information (e.g., prompts)",
    )