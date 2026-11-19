"""CPU-only tests for Llama-3 conversation serialization.

python3 -m unittest discover -s tests -p test_llama3_conversation.py -v
"""

import unittest

from fastchat.conversation import get_conv_template


class Llama3ConversationTest(unittest.TestCase):
    def test_empty_user_message_is_terminated(self):
        conv = get_conv_template("llama-3")
        conv.append_message(conv.roles[0], "")
        conv.append_message(conv.roles[1], None)

        self.assertEqual(
            conv.get_prompt(),
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )

    def test_empty_assistant_message_is_terminated(self):
        conv = get_conv_template("llama-3")
        conv.append_message(conv.roles[0], "Hello")
        conv.append_message(conv.roles[1], "")

        self.assertEqual(
            conv.get_prompt(),
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n<|eot_id|>",
        )

    def test_empty_response_keeps_boundary_before_next_turn(self):
        for system_message in ("", "Synthetic system"):
            with self.subTest(system_message=system_message):
                conv = get_conv_template("llama-3")
                conv.set_system_message(system_message)
                conv.append_message(conv.roles[0], "Hello")
                conv.append_message(conv.roles[1], None)
                conv.update_last_message("")
                conv.append_message(conv.roles[0], "Next question")
                conv.append_message(conv.roles[1], None)
                system_prompt = (
                    "<|start_header_id|>system<|end_header_id|>\n\n"
                    "Synthetic system<|eot_id|>"
                    if system_message
                    else ""
                )

                self.assertEqual(
                    conv.get_prompt(),
                    "<|begin_of_text|>"
                    + system_prompt
                    + "<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"
                    "<|start_header_id|>assistant<|end_header_id|>\n\n<|eot_id|>"
                    "<|start_header_id|>user<|end_header_id|>\n\nNext question<|eot_id|>"
                    "<|start_header_id|>assistant<|end_header_id|>\n\n",
                )

    def test_none_assistant_placeholder_stays_open(self):
        conv = get_conv_template("llama-3")
        conv.append_message(conv.roles[0], "Hello")
        conv.append_message(conv.roles[1], None)

        self.assertEqual(
            conv.get_prompt(),
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )

    def test_whitespace_only_messages_are_still_terminated(self):
        for role in ("user", "assistant"):
            with self.subTest(role=role):
                conv = get_conv_template("llama-3")
                conv.append_message(role, " \t\n")

                self.assertEqual(
                    conv.get_prompt(),
                    "<|begin_of_text|>"
                    f"<|start_header_id|>{role}<|end_header_id|>\n\n<|eot_id|>",
                )

    def test_nonempty_messages_and_system_prompt_are_unchanged(self):
        conv = get_conv_template("llama-3")
        conv.set_system_message("Synthetic system")
        conv.append_message(conv.roles[0], " Hello ")
        conv.append_message(conv.roles[1], " Hi ")
        conv.append_message(conv.roles[0], " Next question ")
        conv.append_message(conv.roles[1], None)

        self.assertEqual(
            conv.get_prompt(),
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\nSynthetic system<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\nHi<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\nNext question<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )

    def test_empty_conversation_is_unchanged(self):
        conv = get_conv_template("llama-3")
        self.assertEqual(conv.get_prompt(), "<|begin_of_text|>")
