import importlib.util
import sys
import types
import unittest
from pathlib import Path

sys.modules.setdefault("transformers", types.ModuleType("transformers"))
tqdm_mod = types.ModuleType("tqdm")
tqdm_mod.tqdm = lambda x, **kwargs: x
sys.modules.setdefault("tqdm", tqdm_mod)

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "fastchat" / "data" / "split_long_conversation.py"
spec = importlib.util.spec_from_file_location("split_long_conversation", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class _Ids:
    def __init__(self, n):
        self.input_ids = [1] * n


def _tokenizer(text):
    return _Ids(int(text))


def _sample(lengths):
    return {
        "id": "a",
        "conversations": [
            {"from": "human" if i % 2 == 0 else "gpt", "value": str(n)}
            for i, n in enumerate(lengths)
        ],
    }


class SplitLongConversationTest(unittest.TestCase):
    def setUp(self):
        mod.tokenizer = _tokenizer

    def test_a_pair_that_still_fits_is_kept_after_a_split(self):
        mod.max_length = 20
        out = mod.split_one_sample(_sample([2, 2, 2, 2]))
        self.assertEqual([len(s["conversations"]) for s in out], [2, 2])

    def test_pairs_that_fit_together_stay_one_sample(self):
        mod.max_length = 40
        out = mod.split_one_sample(_sample([2, 2, 2, 2]))
        self.assertEqual([len(s["conversations"]) for s in out], [4])


if __name__ == "__main__":
    unittest.main()
