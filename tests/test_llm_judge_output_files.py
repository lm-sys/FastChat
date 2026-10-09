"""Offline checks for MT-bench answer and judgment output paths."""

import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from fastchat.conversation import get_conv_template
from fastchat.llm_judge import common, gen_api_answer, gen_model_answer


class TestJudgeOutputFiles(unittest.TestCase):
    """Check bare and nested filenames without running models or remote APIs."""

    def setUp(self):
        resources = ExitStack()
        self.addCleanup(resources.close)
        temp_dir = resources.enter_context(tempfile.TemporaryDirectory())
        resources.callback(os.chdir, os.getcwd())
        os.chdir(temp_dir)
        self.question = {
            "question_id": 1,
            "category": "writing",
            "turns": ["Say hello."],
        }
        self.output_paths = ["results.jsonl", "nested/results.jsonl"]

    def assert_answer_file(self, output_file):
        with open(output_file) as source:
            records = [json.loads(line) for line in source]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["question_id"], 1)
        self.assertEqual(records[0]["choices"], [{"index": 0, "turns": ["Hello."]}])

    def test_api_answers(self):
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    gen_api_answer,
                    "args",
                    SimpleNamespace(force_temperature=None),
                    create=True,
                )
            )
            stack.enter_context(
                patch.object(
                    gen_api_answer, "chat_completion_openai", return_value="Hello."
                )
            )
            for output_file in self.output_paths:
                with self.subTest(output_file=output_file):
                    gen_api_answer.get_answer(
                        self.question, "gpt-3.5-turbo", 1, 16, output_file
                    )
                    gen_model_answer.reorg_answer_file(output_file)
                    self.assert_answer_file(output_file)

    def test_local_model_answers(self):
        model = Mock()
        model.config.is_encoder_decoder = False
        model.generate.return_value = torch.tensor([[1, 2, 3]])
        tokenizer = Mock(return_value=SimpleNamespace(input_ids=[[1, 2]]))
        tokenizer.decode.return_value = "Hello."
        tokenizer.special_tokens_map = {}

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    gen_model_answer, "load_model", return_value=(model, tokenizer)
                )
            )
            stack.enter_context(
                patch.object(
                    gen_model_answer,
                    "get_conversation_template",
                    side_effect=lambda _: get_conv_template("vicuna_v1.1"),
                )
            )
            stack.enter_context(
                patch.object(torch.Tensor, "cuda", lambda tensor: tensor)
            )
            for output_file in self.output_paths:
                with self.subTest(output_file=output_file):
                    gen_model_answer.get_model_answers(
                        model_path="unused",
                        model_id="test-model",
                        questions=[self.question],
                        answer_file=output_file,
                        max_new_token=16,
                        num_choices=1,
                        num_gpus_per_model=1,
                        max_gpu_memory=None,
                        dtype=None,
                        revision="main",
                    )
                    gen_model_answer.reorg_answer_file(output_file)
                    self.assert_answer_file(output_file)

    def test_single_judgments(self):
        judge = common.Judge("test-judge", {"type": "single", "name": "test-prompt"})
        match = common.MatchSingle(self.question, "test-model", {}, judge)
        with patch.object(
            common, "run_judge_single", return_value=(8, "prompt", "judgment")
        ):
            for output_file in self.output_paths:
                with self.subTest(output_file=output_file):
                    result = common.play_a_match_single(match, output_file)
                    with open(output_file) as source:
                        saved = json.load(source)
                    self.assertEqual(saved["question_id"], 1)
                    self.assertEqual(saved["score"], result["score"])
                    self.assertEqual(saved["judgment"], "judgment")

    def test_pairwise_judgments(self):
        judge = common.Judge("test-judge", {"type": "pairwise", "name": "test-prompt"})
        match = common.MatchPair(self.question, "model-a", "model-b", {}, {}, judge)
        with patch.object(
            common, "run_judge_pair", return_value=("A", "prompt", "judgment")
        ):
            for output_file in self.output_paths:
                with self.subTest(output_file=output_file):
                    result = common.play_a_match_pair(match, output_file)
                    with open(output_file) as source:
                        saved = json.load(source)
                    self.assertEqual(saved["question_id"], 1)
                    self.assertEqual(saved["g1_winner"], result["g1_winner"])
                    self.assertEqual(saved["g2_winner"], result["g2_winner"])


if __name__ == "__main__":
    unittest.main()
