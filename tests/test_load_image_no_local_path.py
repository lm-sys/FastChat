"""
python3 -m unittest tests.test_load_image_no_local_path
"""

import tempfile
import unittest
from pathlib import Path

from fastchat.utils import load_image


class LoadImageNoLocalPathTest(unittest.TestCase):
    def test_rejects_local_image_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.png"
            # minimal 1x1 PNG
            path.write_bytes(
                bytes.fromhex(
                    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
                    "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
                )
            )
            with self.assertRaises(ValueError):
                load_image(str(path))


if __name__ == "__main__":
    unittest.main()
