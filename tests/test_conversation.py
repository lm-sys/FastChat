"""
Usage:
python3 -m unittest tests.test_conversation
"""

import unittest

from fastchat.conversation import get_conv_template


class TestYuan2Template(unittest.TestCase):
    def test_message_ending_in_n_is_preserved(self):
        """The YUAN2 template must only strip the trailing ``<n>`` separator,
        not characters (``<``, ``n``, ``>``) that belong to the message."""
        conv = get_conv_template("yuan2")
        conv.append_message(conv.roles[0], "What is the value of n")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        self.assertIn("What is the value of n", prompt)
        self.assertEqual(prompt, "What is the value of n<sep>")

    def test_message_ending_in_angle_bracket_is_preserved(self):
        conv = get_conv_template("yuan2")
        conv.append_message(conv.roles[0], "compare a < b")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        self.assertIn("compare a < b", prompt)

    def test_separator_between_messages_is_kept(self):
        conv = get_conv_template("yuan2")
        conv.append_message(conv.roles[0], "hello")
        conv.append_message(conv.roles[1], "world")
        prompt = conv.get_prompt()
        self.assertEqual(prompt, "hello<n>world<sep>")


if __name__ == "__main__":
    unittest.main()
