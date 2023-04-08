"""
Chat with a model with command line interface.

Usage:
python3 -m fastchat.serve.cli --model ~/model_weights/llama-7b
"""
import argparse
import re

from prompt_toolkit import prompt
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

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
        self._prompt_session = PromptSession(history=InMemoryHistory())
        self._completer = WordCompleter(words=['!exit', '!reset'], pattern=re.compile('$'))
        self._console = Console()

    def prompt_for_input(self, role) -> str:
        self._console.print(f"{role}: ")
        # TODO(suquark): multiline input has some issues. fix it later.
        prompt_input = self._prompt_session.prompt(
            completer=self._completer,
            multiline=False,
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=None)
        self._console.print('\n')
        return prompt_input

    def prompt_for_output(self, role: str):
        self._console.print(f"{role}: ")

    def append_output(self, output: str):
        self._console.print(output, end=" ")

    def finalize_output(self, output: str):
        self._console.print(output, end="\n\n")

    def stream_output(self, role: str, output_stream, skip_echo_len: int):
        """Stream output from a role."""
        self.prompt_for_output(role)
        pre = 0
        # Create a Live context for updating the console output
        with Live(console=self._console, refresh_per_second=4) as live:
            accumulated_text = ""
            
            # Read lines from the stream
            for outputs in output_stream:
                outputs = outputs[skip_echo_len:].strip()
                outputs = outputs.split(" ")
                now = len(outputs) - 1
                if now > pre:
                    accumulated_text += " ".join(outputs[pre:now]) + " "
                    pre = now
                # Render the accumulated text as Markdown
                markdown = Markdown(accumulated_text)
                
                # Update the Live console output
                live.update(markdown)

            accumulated_text += " ".join(outputs[pre:])
            markdown = Markdown(accumulated_text)
            live.update(markdown)

        self._console.print('\n')
        return outputs


def main(args):
    if args.chatio == "simple":
        chatio = SimpleChatIO()
    elif args.chatio == "markdown":
        chatio = MarkdownChatIO()
    else:
        raise ValueError(f"Invalid chatio for console: {args.chatio}")
    try:
        chat_loop(args.model_name, args.device, args.num_gpus, args.load_8bit,
                args.conv_template, args.temperature, args.max_new_tokens,
                chatio, args.debug)
    except KeyboardInterrupt:
        print("exit...")


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
