"""
Chat with a model with command line interface.

Usage:
python3 -m fastchat.serve.cli --model ~/model_weights/llama-7b
"""
import argparse
from fastchat.serve.inference import chat_loop, ChatIO


class SimpleChatIO(ChatIO):
    def prompt_for_input(self, role) -> str:
        return input(f"{role}: ")

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def append_output(self, output: str):
        print(output, end=" ", flush=True)

    def finalize_output(self, output: str):
        print(output, flush=True)


class MarkdownChatIO(ChatIO):
    def __init__(self):
        pass

    def prompt_for_input(self, role) -> str:
        return input(f"{role}: ")

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def append_output(self, output: str):
        print(output, end=" ", flush=True)

    def finalize_output(self, output: str):
        print(output, flush=True)


def main(args):
    if args.chatio == "simple":
        chatio = SimpleChatIO()
    elif args.chatio == "markdown":
        chatio = MarkdownChatIO()
    else:
        raise ValueError(f"Invalid chatio for console: {args.chatio}")
    chat_loop(args.model_name, args.device, args.num_gpus, args.load_8bit,
              args.conv_template, args.temperature, args.max_new_tokens,
              chatio, args.debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--load-8bit", action="store_true",
        help="Use 8-bit quantization.")
    parser.add_argument("--conv-template", type=str, default="v1",
        help="Conversation prompt template.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--chatio", type=str, default="markdown",
                        choices=["simple", "markdown"], help="Chat IO format.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
